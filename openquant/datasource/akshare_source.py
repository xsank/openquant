"""AKShare 数据源实现

基于 akshare 库获取 A 股、港股、美股、基金等多市场行情数据。
"""
from __future__ import annotations

import logging

import akshare as ak
import pandas as pd

from openquant.core.exceptions import DataSourceError
from openquant.core.interfaces import DataSourceInterface
from openquant.core.models import FrequencyType, MarketType

logger = logging.getLogger(__name__)


class AkshareDataSource(DataSourceInterface):
    """AKShare 数据源，支持 A 股、港股、美股、基金等多市场"""

    def get_name(self) -> str:
        return "akshare"

    def get_supported_markets(self) -> list[MarketType]:
        return [MarketType.A_SHARE, MarketType.HK_STOCK, MarketType.US_STOCK, MarketType.FUND]

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        try:
            fetcher = _DAILY_FETCHERS.get(market)
            if fetcher is None:
                raise DataSourceError(f"AKShare 不支持市场类型: {market}")
            return fetcher(symbol, start_date, end_date)
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取日K线失败: {exc}") from exc

    def fetch_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: FrequencyType = FrequencyType.MINUTE_5,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        try:
            period_map = {
                FrequencyType.MINUTE_1: "1",
                FrequencyType.MINUTE_5: "5",
                FrequencyType.MINUTE_15: "15",
                FrequencyType.MINUTE_30: "30",
                FrequencyType.MINUTE_60: "60",
            }
            period = period_map.get(frequency)
            if period is None:
                raise DataSourceError(f"AKShare 不支持分钟频率: {frequency}")

            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            return _normalize_columns(df)
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取分钟K线失败: {exc}") from exc

    def fetch_stock_list(
        self,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        try:
            if market == MarketType.A_SHARE:
                df = ak.stock_zh_a_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.A_SHARE.value
                return df[["symbol", "name", "market"]]
            elif market == MarketType.HK_STOCK:
                df = ak.stock_hk_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.HK_STOCK.value
                return df[["symbol", "name", "market"]]
            elif market == MarketType.US_STOCK:
                df = ak.stock_us_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.US_STOCK.value
                return df[["symbol", "name", "market"]]
            elif market == MarketType.FUND:
                df = ak.fund_etf_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.FUND.value
                return df[["symbol", "name", "market"]]
            else:
                raise DataSourceError(f"AKShare 不支持市场类型: {market}")
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取股票列表失败: {exc}") from exc

    def fetch_realtime_quote(
        self,
        symbol: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> dict:
        try:
            if market == MarketType.A_SHARE:
                df = ak.stock_zh_a_spot_em()
                row = df[df["代码"] == symbol]
                if row.empty:
                    raise DataSourceError(f"未找到股票: {symbol}")
                row = row.iloc[0]
                return {
                    "symbol": symbol,
                    "name": row.get("名称", ""),
                    "price": float(row.get("最新价", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                    "volume": float(row.get("成交量", 0)),
                    "amount": float(row.get("成交额", 0)),
                    "high": float(row.get("最高", 0)),
                    "low": float(row.get("最低", 0)),
                    "open": float(row.get("今开", 0)),
                    "prev_close": float(row.get("昨收", 0)),
                }
            elif market == MarketType.HK_STOCK:
                from datetime import datetime as dt, timedelta
                end = dt.now().strftime("%Y%m%d")
                start = (dt.now() - timedelta(days=7)).strftime("%Y%m%d")
                df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df.empty:
                    raise DataSourceError(f"未获取到港股行情: {symbol}")
                row = df.iloc[-1]
                return {
                    "symbol": symbol,
                    "name": "",
                    "price": float(row.get("收盘", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                    "volume": float(row.get("成交量", 0)),
                    "amount": float(row.get("成交额", 0)),
                    "high": float(row.get("最高", 0)),
                    "low": float(row.get("最低", 0)),
                    "open": float(row.get("开盘", 0)),
                    "prev_close": float(row.get("收盘", 0)),
                }
            else:
                raise DataSourceError(f"实时行情暂不支持市场: {market}")
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取实时行情失败: {exc}") from exc


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为标准格式"""
    column_mapping = {
        "日期": "datetime", "时间": "datetime",
        "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
        "成交量": "volume", "成交额": "amount", "换手率": "turnover_rate",
        "date": "datetime",
    }
    df = df.rename(columns=column_mapping)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _fetch_a_share_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到A股数据，请检查股票代码: {symbol}")
    return _normalize_columns(df)


def _fetch_hk_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_hk_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到港股数据，请检查股票代码: {symbol}")
    return _normalize_columns(df)


def _resolve_us_symbol(symbol: str) -> str:
    """将用户输入的美股代码转换为 AKShare 所需的带交易所前缀格式。

    AKShare 的 stock_us_hist 接口要求 symbol 格式为 "105.GOOGL"（交易所编号.股票代码），
    东方财富使用的交易所编号：105=NASDAQ, 106=NYSE, 107=AMEX。

    解析策略（按优先级）：
    1. 用户已传入带前缀的格式（如 "105.GOOGL"）→ 直接使用
    2. 通过东方财富搜索接口精确匹配股票代码 → 获取正确的 QuoteID
    3. 搜索接口不可用时，依次尝试三大交易所前缀（105/106/107）
    """
    if "." in symbol:
        return symbol

    upper_symbol = symbol.upper()

    quote_id = _search_us_quote_id(upper_symbol)
    if quote_id:
        return quote_id

    logger.warning("无法通过搜索接口解析美股代码 %s，将依次尝试各交易所前缀", upper_symbol)
    for exchange_prefix in _US_EXCHANGE_PREFIXES:
        candidate = f"{exchange_prefix}.{upper_symbol}"
        try:
            df = ak.stock_us_hist(symbol=candidate, period="daily", adjust="qfq")
            if df is not None and not df.empty:
                logger.info("通过尝试确认美股代码: %s", candidate)
                return candidate
        except Exception:
            continue

    raise DataSourceError(
        f"无法解析美股代码: {symbol}，请使用带交易所前缀的格式（如 105.{upper_symbol}）"
    )


# 东方财富交易所编号: 105=NASDAQ, 106=NYSE, 107=AMEX
_US_EXCHANGE_PREFIXES = ("105", "106", "107")


def _search_us_quote_id(symbol: str) -> str | None:
    """通过东方财富搜索接口查找美股的完整 QuoteID。

    该接口轻量快速，无需拉取全量美股列表，返回精确匹配的 QuoteID（如 "105.GOOGL"）。
    """
    import requests

    search_url = "https://searchapi.eastmoney.com/api/suggest/get"
    params = {
        "input": symbol,
        "type": "14",
        "token": "D43BF722C8E33BDC906FB84D85E326E8",
        "count": "10",
    }
    try:
        response = requests.get(search_url, params=params, timeout=10)
        data = response.json()
        suggestions = data.get("QuotationCodeTable", {}).get("Data") or []
        for item in suggestions:
            if item.get("Code", "").upper() == symbol and item.get("SecurityTypeName") == "美股":
                return item["QuoteID"]
    except Exception:
        logger.debug("东方财富搜索接口调用失败，symbol=%s", symbol)
    return None


def _fetch_us_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    full_symbol = _resolve_us_symbol(symbol)
    df = ak.stock_us_hist(
        symbol=full_symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到美股数据，请检查股票代码: {symbol} (解析为 {full_symbol})")
    return _normalize_columns(df)


def _fetch_fund_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到基金数据，请检查基金代码: {symbol}")
    return _normalize_columns(df)


_DAILY_FETCHERS = {
    MarketType.A_SHARE: _fetch_a_share_daily,
    MarketType.HK_STOCK: _fetch_hk_daily,
    MarketType.US_STOCK: _fetch_us_daily,
    MarketType.FUND: _fetch_fund_daily,
}
