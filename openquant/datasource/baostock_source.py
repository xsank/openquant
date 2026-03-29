"""Baostock 数据源实现

基于 baostock 库获取 A 股历史行情数据。
"""
from __future__ import annotations

import logging
from datetime import datetime

import baostock as bs
import pandas as pd

from openquant.core.exceptions import DataSourceError
from openquant.core.interfaces import DataSourceInterface
from openquant.core.models import FrequencyType, MarketType

logger = logging.getLogger(__name__)

_FREQUENCY_MAP = {
    FrequencyType.MINUTE_5: "5",
    FrequencyType.MINUTE_15: "15",
    FrequencyType.MINUTE_30: "30",
    FrequencyType.MINUTE_60: "60",
    FrequencyType.DAILY: "d",
    FrequencyType.WEEKLY: "w",
    FrequencyType.MONTHLY: "m",
}


class BaostockDataSource(DataSourceInterface):
    """Baostock 数据源，支持 A 股日/周/月/分钟 K 线"""

    def __init__(self):
        self._logged_in = False

    def _ensure_login(self) -> None:
        if not self._logged_in:
            result = bs.login()
            if result.error_code != "0":
                raise DataSourceError(f"Baostock 登录失败: {result.error_msg}")
            self._logged_in = True

    def _logout(self) -> None:
        if self._logged_in:
            bs.logout()
            self._logged_in = False

    def get_name(self) -> str:
        return "baostock"

    def get_supported_markets(self) -> list[MarketType]:
        return [MarketType.A_SHARE]

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """将通用代码转为 baostock 格式 (如 600000 -> sh.600000)"""
        code = symbol.replace("sh.", "").replace("sz.", "").replace("SH", "").replace("SZ", "")
        code = code.replace(".", "")
        if code.startswith(("6", "9", "5")):
            return f"sh.{code}"
        return f"sz.{code}"

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        self._ensure_login()
        bs_symbol = self._normalize_symbol(symbol)
        result = bs.query_history_k_data_plus(
            bs_symbol,
            "date,open,high,low,close,volume,amount,turn",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        if result.error_code != "0":
            raise DataSourceError(f"获取日K线失败: {result.error_msg}")

        rows = []
        while result.next():
            rows.append(result.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows,
            columns=["datetime", "open", "high", "low", "close", "volume", "amount", "turnover_rate"],
        )
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def fetch_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: FrequencyType = FrequencyType.MINUTE_5,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        self._ensure_login()
        bs_symbol = self._normalize_symbol(symbol)
        freq_code = _FREQUENCY_MAP.get(frequency)
        if freq_code is None:
            raise DataSourceError(f"Baostock 不支持频率: {frequency}")

        result = bs.query_history_k_data_plus(
            bs_symbol,
            "date,time,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency=freq_code,
            adjustflag="3",
        )
        if result.error_code != "0":
            raise DataSourceError(f"获取分钟K线失败: {result.error_msg}")

        rows = []
        while result.next():
            rows.append(result.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "time", "open", "high", "low", "close", "volume", "amount"])
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"].str[:2] + ":" + df["time"].str[2:4])
        df.drop(columns=["date", "time"], inplace=True)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def fetch_stock_list(
        self,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        self._ensure_login()
        result = bs.query_stock_basic(code_name="", code="")
        if result.error_code != "0":
            raise DataSourceError(f"获取股票列表失败: {result.error_msg}")

        rows = []
        while result.next():
            rows.append(result.get_row_data())

        if not rows:
            return pd.DataFrame(columns=["symbol", "name", "market"])

        df = pd.DataFrame(rows, columns=result.fields)
        df = df.rename(columns={"code": "symbol", "code_name": "name"})
        df["market"] = MarketType.A_SHARE.value
        return df[["symbol", "name", "market"]]

    def fetch_realtime_quote(
        self,
        symbol: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> dict:
        raise DataSourceError("Baostock 不支持实时行情，请使用 akshare 数据源")
