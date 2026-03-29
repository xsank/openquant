"""核心接口定义

定义数据源、存储、策略、引擎的抽象接口，所有实现必须遵循这些接口。
"""
from __future__ import annotations

import abc
from datetime import datetime

import pandas as pd

from openquant.core.models import (
    Bar,
    FrequencyType,
    MarketType,
    Order,
    Portfolio,
    TradeRecord,
)


class DataSourceInterface(abc.ABC):
    """数据源抽象接口

    所有数据源（baostock、akshare 等）必须实现此接口。
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """返回数据源名称"""

    @abc.abstractmethod
    def get_supported_markets(self) -> list[MarketType]:
        """返回支持的市场类型列表"""

    @abc.abstractmethod
    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        """获取日K线数据

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume, amount
        """

    @abc.abstractmethod
    def fetch_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: FrequencyType = FrequencyType.MINUTE_5,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        """获取分钟K线数据"""

    @abc.abstractmethod
    def fetch_stock_list(
        self,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        """获取股票列表

        Returns:
            DataFrame with columns: symbol, name, market
        """

    @abc.abstractmethod
    def fetch_realtime_quote(
        self,
        symbol: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> dict:
        """获取实时行情"""


class StorageInterface(abc.ABC):
    """存储抽象接口

    所有存储后端（SQLite、MySQL 等）必须实现此接口。
    """

    @abc.abstractmethod
    def initialize(self) -> None:
        """初始化存储（建表等）"""

    @abc.abstractmethod
    def save_bars(self, bars: pd.DataFrame, symbol: str, market: MarketType) -> int:
        """保存K线数据，返回写入行数"""

    @abc.abstractmethod
    def load_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
        frequency: FrequencyType = FrequencyType.DAILY,
    ) -> pd.DataFrame:
        """加载K线数据"""

    @abc.abstractmethod
    def save_trade_records(self, records: list[TradeRecord]) -> int:
        """保存交易记录"""

    @abc.abstractmethod
    def load_trade_records(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[TradeRecord]:
        """加载交易记录"""

    @abc.abstractmethod
    def save_portfolio_snapshot(self, portfolio: Portfolio, snapshot_time: datetime) -> None:
        """保存组合快照"""

    @abc.abstractmethod
    def close(self) -> None:
        """关闭存储连接"""


class StrategyInterface(abc.ABC):
    """策略抽象接口

    所有量化策略必须实现此接口。
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """策略名称"""

    @abc.abstractmethod
    def initialize(self, portfolio: Portfolio) -> None:
        """策略初始化，在回测/交易开始前调用"""

    @abc.abstractmethod
    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        """每根K线触发，返回订单列表

        Args:
            bar: 当前K线数据
            portfolio: 当前投资组合状态

        Returns:
            需要执行的订单列表（可为空）
        """

    @abc.abstractmethod
    def on_order_filled(self, order: Order, portfolio: Portfolio) -> None:
        """订单成交回调"""

    def on_finish(self, portfolio: Portfolio) -> None:
        """策略结束回调（可选覆盖）"""


class EngineInterface(abc.ABC):
    """交易引擎抽象接口"""

    @abc.abstractmethod
    def set_strategy(self, strategy: StrategyInterface) -> None:
        """设置交易策略"""

    @abc.abstractmethod
    def run(self) -> Portfolio:
        """运行引擎，返回最终组合状态"""

    @abc.abstractmethod
    def get_results(self) -> dict:
        """获取运行结果统计"""
