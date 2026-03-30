"""YFinance 事件数据源实现

基于 yfinance 库获取港股、美股的事件因子数据，包括：
- 财报发布日期（含 EPS 预期/实际对比）
- 分红派息历史
- 拆股/合股历史
- 分析师评级变动
- 机构持仓变动

yfinance 是全球最流行的开源金融数据库，原生支持港股（.HK）和美股，
弥补了 akshare 在港股/美股事件数据上的不足。
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from openquant.core.interfaces import EventSourceInterface
from openquant.core.models import (
    EventFactor,
    EventSentiment,
    EventType,
    MarketType,
)

logger = logging.getLogger(__name__)


class YFinanceEventSource(EventSourceInterface):
    """YFinance 事件数据源

    通过 yfinance 获取港股和美股的事件因子数据。
    也支持 A 股（通过 .SS/.SZ 后缀），但 A 股建议优先使用 AkshareEventSource。
    """

    def get_name(self) -> str:
        return "yfinance_event"

    def get_supported_event_types(self) -> list[EventType]:
        return [
            EventType.EARNINGS,
            EventType.DIVIDEND,
            EventType.ANALYST_UPGRADE,
            EventType.ANALYST_DOWNGRADE,
            EventType.INSIDER,
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

        yf_symbol = _convert_to_yfinance_symbol(symbol, market)
        logger.info("yfinance 获取事件: %s -> %s", symbol, yf_symbol)

        try:
            ticker = yf.Ticker(yf_symbol)
        except Exception as exc:
            logger.warning("yfinance 创建 Ticker 失败: %s - %s", yf_symbol, exc)
            return []

        all_events: list[EventFactor] = []
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        fetcher_map = {
            EventType.EARNINGS: self._fetch_earnings,
            EventType.DIVIDEND: self._fetch_dividends,
            EventType.ANALYST_UPGRADE: self._fetch_recommendations,
            EventType.ANALYST_DOWNGRADE: self._fetch_recommendations,
            EventType.INSIDER: self._fetch_insider_transactions,
        }

        fetched_types: set[str] = set()
        for event_type in event_types:
            fetcher = fetcher_map.get(event_type)
            if fetcher is None:
                continue

            fetcher_key = fetcher.__name__
            if fetcher_key in fetched_types:
                continue
            fetched_types.add(fetcher_key)

            try:
                events = fetcher(ticker, symbol, market)
                filtered = [
                    e for e in events
                    if start_dt <= pd.Timestamp(e.event_date) <= end_dt
                ]
                all_events.extend(filtered)
            except Exception as exc:
                logger.warning("yfinance 获取 %s 的 %s 事件失败: %s", symbol, event_type.value, exc)

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
        start_date = (datetime.now() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        events = self.fetch_events(symbol, start_date, end_date, event_types, market)
        events.sort(key=lambda e: e.event_date, reverse=True)
        return events[:limit]

    def _fetch_earnings(
        self,
        ticker: yf.Ticker,
        symbol: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取财报发布事件（含 EPS 预期/实际对比）"""
        events: list[EventFactor] = []

        try:
            earnings_dates = ticker.get_earnings_dates(limit=20)
            if earnings_dates is None or earnings_dates.empty:
                return events

            for idx, row in earnings_dates.iterrows():
                try:
                    event_date = pd.Timestamp(idx).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                eps_estimate = _safe_float(row.get("EPS Estimate"))
                eps_actual = _safe_float(row.get("Reported EPS"))
                surprise_pct = _safe_float(row.get("Surprise(%)"))

                if pd.isna(eps_actual) or eps_actual == 0:
                    sentiment = EventSentiment.NEUTRAL
                    strength = 0.3
                    description = f"财报发布日 (EPS预期: {eps_estimate:.2f})"
                elif surprise_pct > 10:
                    sentiment = EventSentiment.BULLISH
                    strength = min(surprise_pct / 30, 2.0)
                    description = f"财报超预期: EPS={eps_actual:.2f} vs 预期={eps_estimate:.2f}, 超预期{surprise_pct:.1f}%"
                elif surprise_pct < -10:
                    sentiment = EventSentiment.BEARISH
                    strength = min(abs(surprise_pct) / 30, 2.0)
                    description = f"财报不及预期: EPS={eps_actual:.2f} vs 预期={eps_estimate:.2f}, 低于预期{abs(surprise_pct):.1f}%"
                else:
                    sentiment = EventSentiment.NEUTRAL
                    strength = 0.3
                    description = f"财报符合预期: EPS={eps_actual:.2f} vs 预期={eps_estimate:.2f}"

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.EARNINGS,
                    sentiment=sentiment,
                    strength=strength,
                    description=description,
                    source="yfinance_earnings",
                    market=market,
                    extra={
                        "eps_estimate": eps_estimate,
                        "eps_actual": eps_actual,
                        "surprise_pct": surprise_pct,
                    },
                ))
        except Exception as exc:
            logger.debug("yfinance 获取财报日期失败: %s", exc)

        return events

    def _fetch_dividends(
        self,
        ticker: yf.Ticker,
        symbol: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取分红派息事件"""
        events: list[EventFactor] = []

        try:
            dividends = ticker.dividends
            if dividends is None or dividends.empty:
                return events

            for date_idx, amount in dividends.items():
                try:
                    event_date = pd.Timestamp(date_idx).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                dividend_amount = float(amount)
                if dividend_amount <= 0:
                    continue

                info = ticker.info or {}
                current_price = _safe_float(info.get("currentPrice", info.get("regularMarketPrice", 0)))

                if current_price > 0:
                    dividend_yield = dividend_amount / current_price
                    if dividend_yield > 0.02:
                        sentiment = EventSentiment.BULLISH
                        strength = min(dividend_yield * 20, 1.5)
                    else:
                        sentiment = EventSentiment.BULLISH
                        strength = 0.5
                else:
                    sentiment = EventSentiment.BULLISH
                    strength = 0.5

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.DIVIDEND,
                    sentiment=sentiment,
                    strength=strength,
                    description=f"分红派息: 每股 {dividend_amount:.4f}",
                    source="yfinance_dividends",
                    market=market,
                    extra={"dividend_amount": dividend_amount},
                ))
        except Exception as exc:
            logger.debug("yfinance 获取分红数据失败: %s", exc)

        return events

    def _fetch_recommendations(
        self,
        ticker: yf.Ticker,
        symbol: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取分析师评级变动事件"""
        events: list[EventFactor] = []

        try:
            recommendations = ticker.recommendations
            if recommendations is None or recommendations.empty:
                return events

            for idx, row in recommendations.iterrows():
                try:
                    event_date = pd.Timestamp(idx).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                firm = str(row.get("Firm", ""))
                to_grade = str(row.get("To Grade", "")).lower()
                from_grade = str(row.get("From Grade", "")).lower()
                action = str(row.get("Action", "")).lower()

                bullish_keywords = {"buy", "overweight", "outperform", "strong buy", "positive", "accumulate"}
                bearish_keywords = {"sell", "underweight", "underperform", "reduce", "negative"}

                if action in ("up", "upgrade", "init") and any(kw in to_grade for kw in bullish_keywords):
                    sentiment = EventSentiment.BULLISH
                    event_type = EventType.ANALYST_UPGRADE
                    strength = 0.8
                    description = f"{firm}: 上调评级至 {to_grade}"
                elif action in ("down", "downgrade") or any(kw in to_grade for kw in bearish_keywords):
                    sentiment = EventSentiment.BEARISH
                    event_type = EventType.ANALYST_DOWNGRADE
                    strength = 0.8
                    description = f"{firm}: 下调评级至 {to_grade}"
                elif any(kw in to_grade for kw in bullish_keywords):
                    sentiment = EventSentiment.BULLISH
                    event_type = EventType.ANALYST_UPGRADE
                    strength = 0.5
                    description = f"{firm}: 维持 {to_grade} 评级"
                elif any(kw in to_grade for kw in bearish_keywords):
                    sentiment = EventSentiment.BEARISH
                    event_type = EventType.ANALYST_DOWNGRADE
                    strength = 0.5
                    description = f"{firm}: 维持 {to_grade} 评级"
                else:
                    sentiment = EventSentiment.NEUTRAL
                    event_type = EventType.ANALYST_UPGRADE
                    strength = 0.2
                    description = f"{firm}: {to_grade}"

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=event_type,
                    sentiment=sentiment,
                    strength=strength,
                    description=description,
                    source="yfinance_recommendations",
                    market=market,
                    extra={
                        "firm": firm,
                        "to_grade": to_grade,
                        "from_grade": from_grade,
                        "action": action,
                    },
                ))
        except Exception as exc:
            logger.debug("yfinance 获取分析师评级失败: %s", exc)

        return events

    def _fetch_insider_transactions(
        self,
        ticker: yf.Ticker,
        symbol: str,
        market: MarketType,
    ) -> list[EventFactor]:
        """获取内部人交易事件"""
        events: list[EventFactor] = []

        try:
            insider_transactions = ticker.insider_transactions
            if insider_transactions is None or insider_transactions.empty:
                return events

            for _, row in insider_transactions.iterrows():
                date_str = str(row.get("Start Date", row.get("Date", "")))
                if not date_str or date_str == "nan":
                    continue

                try:
                    event_date = pd.Timestamp(date_str).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                insider_name = str(row.get("Insider", row.get("Name", "")))
                transaction = str(row.get("Transaction", "")).lower()
                shares = _safe_float(row.get("Shares", 0))
                value = _safe_float(row.get("Value", 0))

                if "buy" in transaction or "purchase" in transaction:
                    sentiment = EventSentiment.BULLISH
                    strength = min(abs(value) / 1e7, 1.5) if value > 0 else 0.6
                    description = f"内部人买入: {insider_name}, {int(shares)}股, ${value:,.0f}"
                elif "sell" in transaction or "sale" in transaction:
                    sentiment = EventSentiment.BEARISH
                    strength = min(abs(value) / 1e7, 1.5) if value > 0 else 0.6
                    description = f"内部人卖出: {insider_name}, {int(shares)}股, ${value:,.0f}"
                else:
                    sentiment = EventSentiment.NEUTRAL
                    strength = 0.2
                    description = f"内部人交易: {insider_name}, {transaction}"

                events.append(EventFactor(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=EventType.INSIDER,
                    sentiment=sentiment,
                    strength=strength,
                    description=description,
                    source="yfinance_insider",
                    market=market,
                    extra={
                        "insider": insider_name,
                        "transaction": transaction,
                        "shares": shares,
                        "value": value,
                    },
                ))
        except Exception as exc:
            logger.debug("yfinance 获取内部人交易失败: %s", exc)

        return events


def _convert_to_yfinance_symbol(symbol: str, market: MarketType) -> str:
    """将 openquant 的 symbol 格式转换为 yfinance 格式

    转换规则：
    - 港股: "09988" -> "9988.HK"
    - 美股: "105.BABA" -> "BABA", "BABA" -> "BABA"
    - A 股: "600519" -> "600519.SS" (上海), "000001" -> "000001.SZ" (深圳)
    """
    if market == MarketType.HK_STOCK:
        clean_symbol = symbol.lstrip("0")
        if not clean_symbol:
            clean_symbol = "0"
        return f"{clean_symbol}.HK"

    elif market == MarketType.US_STOCK:
        if "." in symbol:
            return symbol.split(".", 1)[1]
        return symbol

    elif market == MarketType.A_SHARE:
        if symbol.startswith(("6", "9")):
            return f"{symbol}.SS"
        else:
            return f"{symbol}.SZ"

    return symbol


def _safe_float(value, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default
