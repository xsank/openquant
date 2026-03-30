"""聚合事件数据源

自动按市场类型路由到最佳的事件数据源：
- A 股 → AkshareEventSource（akshare 覆盖最全）
- 港股 → YFinanceEventSource（yfinance 原生支持 .HK）
- 美股 → YFinanceEventSource（yfinance 原生支持美股）

对外提供统一的 EventSourceInterface 接口，使用方无需关心底层数据源差异。
"""
from __future__ import annotations

import logging

from openquant.core.interfaces import EventSourceInterface
from openquant.core.models import (
    EventFactor,
    EventType,
    MarketType,
)

logger = logging.getLogger(__name__)


class CompositeEventSource(EventSourceInterface):
    """聚合事件数据源

    根据市场类型自动选择最佳的底层事件数据源。
    支持按优先级回退：如果首选数据源获取失败，自动尝试备选数据源。
    """

    def __init__(self):
        self._sources: dict[MarketType, list[EventSourceInterface]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册默认的数据源路由"""
        try:
            from openquant.datasource.akshare_event_source import AkshareEventSource
            akshare_source = AkshareEventSource()
        except ImportError:
            akshare_source = None
            logger.warning("akshare 事件数据源不可用")

        try:
            from openquant.datasource.yfinance_event_source import YFinanceEventSource
            yfinance_source = YFinanceEventSource()
        except ImportError:
            yfinance_source = None
            logger.warning("yfinance 事件数据源不可用")

        if akshare_source:
            self._sources[MarketType.A_SHARE] = [akshare_source]
            if yfinance_source:
                self._sources[MarketType.A_SHARE].append(yfinance_source)

        if yfinance_source:
            self._sources[MarketType.HK_STOCK] = [yfinance_source]
            self._sources[MarketType.US_STOCK] = [yfinance_source]

            if akshare_source:
                self._sources.setdefault(MarketType.HK_STOCK, []).append(akshare_source)

    def get_name(self) -> str:
        return "composite_event"

    def get_supported_event_types(self) -> list[EventType]:
        all_types: set[EventType] = set()
        for sources in self._sources.values():
            for source in sources:
                all_types.update(source.get_supported_event_types())
        return list(all_types)

    def fetch_events(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        event_types: list[EventType] | None = None,
        market: MarketType = MarketType.A_SHARE,
    ) -> list[EventFactor]:
        sources = self._sources.get(market, [])
        if not sources:
            logger.warning("没有可用的事件数据源支持市场类型: %s", market.value)
            return []

        for source in sources:
            try:
                events = source.fetch_events(symbol, start_date, end_date, event_types, market)
                if events:
                    logger.info(
                        "通过 %s 获取到 %s 的 %d 个事件因子",
                        source.get_name(), symbol, len(events),
                    )
                    return events
                logger.debug("%s 未返回 %s 的事件数据，尝试下一个数据源", source.get_name(), symbol)
            except Exception as exc:
                logger.warning(
                    "%s 获取 %s 事件失败: %s，尝试下一个数据源",
                    source.get_name(), symbol, exc,
                )

        logger.info("所有数据源均未获取到 %s 的事件数据", symbol)
        return []

    def fetch_latest_events(
        self,
        symbol: str,
        event_types: list[EventType] | None = None,
        market: MarketType = MarketType.A_SHARE,
        limit: int = 10,
    ) -> list[EventFactor]:
        sources = self._sources.get(market, [])
        if not sources:
            return []

        for source in sources:
            try:
                events = source.fetch_latest_events(symbol, event_types, market, limit)
                if events:
                    return events
            except Exception as exc:
                logger.warning("%s 获取最新事件失败: %s", source.get_name(), exc)

        return []

    def get_source_for_market(self, market: MarketType) -> EventSourceInterface | None:
        """获取指定市场的首选事件数据源"""
        sources = self._sources.get(market, [])
        return sources[0] if sources else None

    def list_market_sources(self) -> dict[str, list[str]]:
        """列出各市场的数据源配置"""
        return {
            market.value: [s.get_name() for s in sources]
            for market, sources in self._sources.items()
        }
