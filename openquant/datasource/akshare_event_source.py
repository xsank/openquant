"""AKShare 事件数据源实现

基于 akshare 库获取 A 股、港股的事件因子数据，包括：
- 财报发布（业绩预告/快报）
- 分红派息
- 大宗交易
- 股东增减持
- 股票回购
"""
from __future__ import annotations

import logging
from datetime import datetime

import akshare as ak
import pandas as pd

from openquant.core.exceptions import DataSourceError
from openquant.core.interfaces import EventSourceInterface
from openquant.core.models import (
    EventFactor,
    EventSentiment,
    EventType,
    MarketType,
)

logger = logging.getLogger(__name__)


class AkshareEventSource(EventSourceInterface):
    """AKShare 事件数据源

    通过 akshare 获取 A 股的各类事件因子数据。
    港股/美股事件数据有限，部分事件类型可能不可用。
    """

    def get_name(self) -> str:
        return "akshare_event"

    def get_supported_event_types(self) -> list[EventType]:
        return [
            EventType.EARNINGS,
            EventType.DIVIDEND,
            EventType.BLOCK_TRADE,
            EventType.SHAREHOLDER,
            EventType.BUYBACK,
        ]

    def fetch_events(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        event_types: list[EventType] | None = None,
        market: MarketType = MarketType.A_SHARE,
    ) -> list[EventFactor]:
        if event_types is None:
            event_types = self.get_supported_event_types()

        all_events: list[EventFactor] = []
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        fetcher_map = {
            EventType.EARNINGS: self._fetch_earnings,
            EventType.DIVIDEND: self._fetch_dividends,
            EventType.BLOCK_TRADE: self._fetch_block_trades,
            EventType.SHAREHOLDER: self._fetch_shareholder_changes,
            EventType.BUYBACK: self._fetch_buybacks,
        }

        for event_type in event_types:
            fetcher = fetcher_map.get(event_type)
            if fetcher is None:
                logger.debug("不支持的事件类型: %s，跳过", event_type)
                continue
            try:
                events = fetcher(symbol, start_date, end_date, market)
                filtered = [
                    e for e in events
                    if start_dt <= pd.Timestamp(e.event_date) <= end_dt
                ]
                all_events.extend(filtered)
            except Exception as exc:
                logger.warning("获取 %s 的 %s 事件失败: %s", symbol, event_type.value, exc)

        all_events.sort(key=lambda e: e.event_date)
        return all_events

    def fetch_latest_events(
        self,
        symbol: str,
        event_types: list[EventType] | None = None,
        market: MarketType = MarketType.A_SHARE,
        limit: int = 10,
    ) -> list[EventFactor]:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        events = self.fetch_events(symbol, start_date, end_date, event_types, market)
        events.sort(key=lambda e: e.event_date, reverse=True)
        return events[:limit]

    def _fetch_earnings(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取业绩预告/快报事件"""
        events: list[EventFactor] = []
        if market != MarketType.A_SHARE:
            return events

        try:
            df = ak.stock_yjbb_em(date="")
            if df is None or df.empty:
                return events

            symbol_col = "股票代码" if "股票代码" in df.columns else None
            if symbol_col is None:
                return events

            df_symbol = df[df[symbol_col] == symbol]
            if df_symbol.empty:
                return events

            for _, row in df_symbol.iterrows():
                report_date_str = str(row.get("公告日期", row.get("报告日期", "")))
                if not report_date_str or report_date_str == "nan":
                    continue

                try:
                    event_date = pd.Timestamp(report_date_str).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                net_profit_growth = _safe_float(row.get("净利润同比增长", row.get("净利润同比", 0)))
                revenue_growth = _safe_float(row.get("营业收入同比增长", row.get("营收同比", 0)))

                if net_profit_growth > 30:
                    sentiment = EventSentiment.BULLISH
                    strength = min(net_profit_growth / 100, 2.0)
                elif net_profit_growth < -20:
                    sentiment = EventSentiment.BEARISH
                    strength = min(abs(net_profit_growth) / 100, 2.0)
                else:
                    sentiment = EventSentiment.NEUTRAL
                    strength = 0.3

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.EARNINGS,
                    sentiment=sentiment,
                    strength=strength,
                    description=f"净利润同比: {net_profit_growth:.1f}%, 营收同比: {revenue_growth:.1f}%",
                    source="akshare_yjbb",
                    market=market,
                    extra={"net_profit_growth": net_profit_growth, "revenue_growth": revenue_growth},
                ))
        except Exception as exc:
            logger.debug("获取业绩快报失败: %s", exc)

        return events

    def _fetch_dividends(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取分红派息事件"""
        events: list[EventFactor] = []
        if market != MarketType.A_SHARE:
            return events

        try:
            df = ak.stock_fhps_detail_em(symbol=symbol)
            if df is None or df.empty:
                return events

            for _, row in df.iterrows():
                ex_date_str = str(row.get("除权除息日", ""))
                if not ex_date_str or ex_date_str == "nan" or ex_date_str == "--":
                    continue

                try:
                    event_date = pd.Timestamp(ex_date_str).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                cash_dividend = _safe_float(row.get("每股派息", row.get("派息", 0)))
                bonus_shares = _safe_float(row.get("每股送股", row.get("送股", 0)))
                transfer_shares = _safe_float(row.get("每股转增", row.get("转增", 0)))

                total_return = cash_dividend + (bonus_shares + transfer_shares) * 0.5

                if total_return > 0.5:
                    sentiment = EventSentiment.BULLISH
                    strength = min(total_return / 2.0, 1.5)
                elif total_return > 0:
                    sentiment = EventSentiment.BULLISH
                    strength = 0.5
                else:
                    sentiment = EventSentiment.NEUTRAL
                    strength = 0.1

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.DIVIDEND,
                    sentiment=sentiment,
                    strength=strength,
                    description=f"派息: {cash_dividend}, 送股: {bonus_shares}, 转增: {transfer_shares}",
                    source="akshare_fhps",
                    market=market,
                    extra={
                        "cash_dividend": cash_dividend,
                        "bonus_shares": bonus_shares,
                        "transfer_shares": transfer_shares,
                    },
                ))
        except Exception as exc:
            logger.debug("获取分红数据失败: %s", exc)

        return events

    def _fetch_block_trades(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取大宗交易事件"""
        events: list[EventFactor] = []
        if market != MarketType.A_SHARE:
            return events

        try:
            df = ak.stock_dzjy_mrmx(symbol=symbol, start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""))
            if df is None or df.empty:
                return events

            for _, row in df.iterrows():
                trade_date_str = str(row.get("交易日期", ""))
                if not trade_date_str or trade_date_str == "nan":
                    continue

                try:
                    event_date = pd.Timestamp(trade_date_str).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                trade_price = _safe_float(row.get("成交价", 0))
                close_price = _safe_float(row.get("收盘价", 0))
                volume = _safe_float(row.get("成交量", row.get("成交数量", 0)))

                if close_price > 0 and trade_price > 0:
                    premium_rate = (trade_price - close_price) / close_price
                    if premium_rate > 0.02:
                        sentiment = EventSentiment.BULLISH
                        strength = min(premium_rate * 10, 1.5)
                    elif premium_rate < -0.05:
                        sentiment = EventSentiment.BEARISH
                        strength = min(abs(premium_rate) * 5, 1.5)
                    else:
                        sentiment = EventSentiment.NEUTRAL
                        strength = 0.3
                else:
                    sentiment = EventSentiment.NEUTRAL
                    strength = 0.3

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.BLOCK_TRADE,
                    sentiment=sentiment,
                    strength=strength,
                    description=f"大宗交易: 价格={trade_price}, 量={volume}",
                    source="akshare_dzjy",
                    market=market,
                    extra={"trade_price": trade_price, "volume": volume},
                ))
        except Exception as exc:
            logger.debug("获取大宗交易数据失败: %s", exc)

        return events

    def _fetch_shareholder_changes(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取股东增减持事件"""
        events: list[EventFactor] = []
        if market != MarketType.A_SHARE:
            return events

        try:
            df = ak.stock_gpzy_pledge_ratio_detail_em(symbol=symbol)
            if df is None or df.empty:
                return events

            for _, row in df.iterrows():
                date_str = str(row.get("公告日期", ""))
                if not date_str or date_str == "nan":
                    continue

                try:
                    event_date = pd.Timestamp(date_str).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                pledge_ratio = _safe_float(row.get("质押比例", 0))

                if pledge_ratio > 50:
                    sentiment = EventSentiment.BEARISH
                    strength = min(pledge_ratio / 100, 1.5)
                elif pledge_ratio > 30:
                    sentiment = EventSentiment.BEARISH
                    strength = 0.5
                else:
                    sentiment = EventSentiment.NEUTRAL
                    strength = 0.2

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.SHAREHOLDER,
                    sentiment=sentiment,
                    strength=strength,
                    description=f"股权质押比例: {pledge_ratio:.1f}%",
                    source="akshare_gpzy",
                    market=market,
                    extra={"pledge_ratio": pledge_ratio},
                ))
        except Exception as exc:
            logger.debug("获取股东变动数据失败: %s", exc)

        return events

    def _fetch_buybacks(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取股票回购事件"""
        events: list[EventFactor] = []
        if market != MarketType.A_SHARE:
            return events

        try:
            df = ak.stock_repurchase_em()
            if df is None or df.empty:
                return events

            symbol_col = None
            for col_name in ["代码", "股票代码"]:
                if col_name in df.columns:
                    symbol_col = col_name
                    break
            if symbol_col is None:
                return events

            df_symbol = df[df[symbol_col] == symbol]
            if df_symbol.empty:
                return events

            for _, row in df_symbol.iterrows():
                date_str = str(row.get("公告日期", row.get("董事会预案公告日", "")))
                if not date_str or date_str == "nan":
                    continue

                try:
                    event_date = pd.Timestamp(date_str).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                amount = _safe_float(row.get("回购金额", row.get("已回购金额", 0)))

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.BUYBACK,
                    sentiment=EventSentiment.BULLISH,
                    strength=min(1.0, amount / 1e9) if amount > 0 else 0.5,
                    description=f"股票回购: 金额={amount/1e4:.0f}万",
                    source="akshare_repurchase",
                    market=market,
                    extra={"amount": amount},
                ))
        except Exception as exc:
            logger.debug("获取回购数据失败: %s", exc)

        return events


def _safe_float(value, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    try:
        if value is None or str(value) in ("", "nan", "--", "None"):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default
