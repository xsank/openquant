"""YFinance 数据源实现

基于 yfinance 库获取美股、港股行情数据，作为东方财富(akshare)的备用数据源。
支持与 akshare 相同的缓存目录和数据格式，确保数据可互通。
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from openquant.core.exceptions import DataSourceError
from openquant.core.interfaces import DataSourceInterface
from openquant.core.models import FrequencyType, MarketType
from openquant.datasource.retry import get_global_throttle

logger = logging.getLogger(__name__)

# 必须包含的列（与 akshare_source 保持一致）
_REQUIRED_COLUMNS = {"datetime", "open", "high", "low", "close", "volume"}

# 东方财富代码 → yfinance ticker 的映射规则
# 105.XXXX (NASDAQ) / 106.XXXX (NYSE) → 直接用 XXXX
# 港股 XXXXX → XXXXX.HK
_HK_SUFFIX = ".HK"


def _eastmoney_to_yfinance_ticker(symbol: str, market: MarketType) -> str:
    """将东方财富格式的代码转为 yfinance ticker"""
    if market == MarketType.US_STOCK:
        if "." in symbol:
            return symbol.split(".", 1)[1]
        return symbol
    elif market == MarketType.HK_STOCK:
        code = symbol.zfill(4)
        return f"{code}{_HK_SUFFIX}"
    elif market == MarketType.A_SHARE:
        suffix = ".SS" if symbol.startswith("6") else ".SZ"
        return f"{symbol}{suffix}"
    else:
        raise DataSourceError(f"YFinance 不支持市场类型: {market}")


def _normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """将 yfinance 返回的 DataFrame 标准化为系统统一格式"""
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    # yfinance 返回的列名是 Open/High/Low/Close/Volume（首字母大写）
    column_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    result = result.rename(columns=column_mapping)

    # 索引是日期，转为 datetime 列
    if result.index.name == "Date" or result.index.name == "Datetime":
        result = result.reset_index()
        result = result.rename(columns={"Date": "datetime", "Datetime": "datetime"})
    elif "Date" in result.columns:
        result = result.rename(columns={"Date": "datetime"})

    if "datetime" not in result.columns and result.index.dtype == "datetime64[ns]":
        result = result.reset_index()
        result.columns = ["datetime"] + list(result.columns[1:])

    if "datetime" in result.columns:
        result["datetime"] = pd.to_datetime(result["datetime"]).dt.tz_localize(None)

    # 确保数值列正确
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for col in numeric_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    # 添加 amount 列（yfinance 不提供成交额，用 close*volume 近似）
    if "amount" not in result.columns and "close" in result.columns and "volume" in result.columns:
        result["amount"] = result["close"] * result["volume"]

    # 完整性校验
    missing = _REQUIRED_COLUMNS - set(result.columns)
    if missing:
        logger.warning("YFinance 数据缺少列: %s", missing)
        return pd.DataFrame()

    # 过滤掉全零行（yfinance 有时会返回无效数据）
    valid_mask = result["close"] > 0
    result = result[valid_mask].reset_index(drop=True)

    return result


class YFinanceDataSource(DataSourceInterface):
    """YFinance 数据源，支持美股、港股

    作为 akshare（东方财富）的备用数据源，当东方财富 API 限流时自动降级使用。
    数据格式与 akshare 完全兼容，可共用同一缓存目录。
    """

    def get_name(self) -> str:
        return "yfinance"

    def get_supported_markets(self) -> list[MarketType]:
        return [MarketType.US_STOCK, MarketType.HK_STOCK, MarketType.A_SHARE]

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        throttle = get_global_throttle()
        ticker = _eastmoney_to_yfinance_ticker(symbol, market)

        try:
            throttle.wait()
            stock = yf.Ticker(ticker)
            # yfinance 的 end 是 exclusive，需要加一天
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            df = stock.history(
                start=start_date,
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            throttle.mark()

            if df is None or df.empty:
                raise DataSourceError(
                    f"YFinance 未获取到数据: {ticker} ({start_date} ~ {end_date})"
                )

            normalized = _normalize_yfinance_df(df)
            if normalized.empty:
                raise DataSourceError(
                    f"YFinance 数据标准化后为空: {ticker}"
                )

            logger.info(
                "YFinance 获取成功: %s → %s, %d 条数据",
                symbol, ticker, len(normalized),
            )
            return normalized

        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(
                f"YFinance 获取日K线失败: {ticker} - {exc}"
            ) from exc

    def fetch_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: FrequencyType = FrequencyType.MINUTE_5,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        raise DataSourceError("YFinance 暂不支持分钟K线获取")

    def fetch_stock_list(
        self,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        raise DataSourceError("YFinance 暂不支持股票列表获取")

    def fetch_realtime_quote(
        self,
        symbol: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> dict:
        throttle = get_global_throttle()
        ticker = _eastmoney_to_yfinance_ticker(symbol, market)

        try:
            throttle.wait()
            stock = yf.Ticker(ticker)
            info = stock.info
            throttle.mark()

            return {
                "symbol": symbol,
                "name": info.get("shortName", ""),
                "price": float(info.get("regularMarketPrice", info.get("previousClose", 0))),
                "change_pct": float(info.get("regularMarketChangePercent", 0)) * 100,
                "volume": float(info.get("regularMarketVolume", 0)),
                "amount": 0.0,
                "high": float(info.get("regularMarketDayHigh", 0)),
                "low": float(info.get("regularMarketDayLow", 0)),
                "open": float(info.get("regularMarketOpen", 0)),
                "prev_close": float(info.get("regularMarketPreviousClose", 0)),
            }
        except Exception as exc:
            raise DataSourceError(
                f"YFinance 获取实时行情失败: {ticker} - {exc}"
            ) from exc
