"""核心接口定义

定义数据源、存储、策略、引擎的抽象接口，所有实现必须遵循这些接口。
"""
from __future__ import annotations

import abc
from datetime import datetime

import pandas as pd

from openquant.core.models import (
    Bar,
    EventFactor,
    EventType,
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


class EventSourceInterface(abc.ABC):
    """事件数据源抽象接口

    所有事件数据源必须实现此接口，用于获取影响股价的离散事件数据，
    如财报发布、分红派息、大宗交易、股东增减持、新闻舆情等。
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """返回事件数据源名称"""

    @abc.abstractmethod
    def get_supported_event_types(self) -> list[EventType]:
        """返回支持的事件类型列表"""

    @abc.abstractmethod
    def fetch_events(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        event_types: list[EventType] | None = None,
        market: MarketType = MarketType.A_SHARE,
    ) -> list[EventFactor]:
        """获取指定标的在时间范围内的事件列表

        Args:
            symbol: 标的代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            event_types: 要获取的事件类型列表，None 表示获取所有类型
            market: 市场类型

        Returns:
            事件因子列表，按事件日期排序
        """

    @abc.abstractmethod
    def fetch_latest_events(
        self,
        symbol: str,
        event_types: list[EventType] | None = None,
        market: MarketType = MarketType.A_SHARE,
        limit: int = 10,
    ) -> list[EventFactor]:
        """获取最新的事件列表（用于实时/模拟交易）

        Args:
            symbol: 标的代码
            event_types: 要获取的事件类型列表
            market: 市场类型
            limit: 最大返回数量

        Returns:
            事件因子列表，按事件日期倒序
        """


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
