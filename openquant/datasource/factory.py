"""数据源工厂

通过注册机制管理数据源，支持按名称或市场类型获取数据源实例。
"""
from __future__ import annotations

from openquant.core.exceptions import DataSourceError
from openquant.core.interfaces import DataSourceInterface
from openquant.core.models import MarketType


class DataSourceFactory:
    """数据源工厂，支持注册和获取数据源"""

    _registry: dict[str, type[DataSourceInterface]] = {}
    _instances: dict[str, DataSourceInterface] = {}

    @classmethod
    def register(cls, name: str, source_class: type[DataSourceInterface]) -> None:
        """注册数据源类"""
        cls._registry[name] = source_class

    @classmethod
    def get(cls, name: str) -> DataSourceInterface:
        """按名称获取数据源实例（单例）"""
        if name not in cls._instances:
            source_class = cls._registry.get(name)
            if source_class is None:
                raise DataSourceError(
                    f"未注册的数据源: {name}，可用数据源: {list(cls._registry.keys())}"
                )
            cls._instances[name] = source_class()
        return cls._instances[name]

    @classmethod
    def get_by_market(cls, market: MarketType) -> DataSourceInterface:
        """按市场类型获取第一个支持该市场的数据源"""
        for name in cls._registry:
            instance = cls.get(name)
            if market in instance.get_supported_markets():
                return instance
        raise DataSourceError(f"没有数据源支持市场类型: {market}")

    @classmethod
    def list_sources(cls) -> list[str]:
        """列出所有已注册的数据源名称"""
        return list(cls._registry.keys())

    @classmethod
    def register_defaults(cls) -> None:
        """注册默认数据源"""
        from openquant.datasource.baostock_source import BaostockDataSource
        from openquant.datasource.akshare_source import AkshareDataSource
        cls.register("baostock", BaostockDataSource)
        cls.register("akshare", AkshareDataSource)
