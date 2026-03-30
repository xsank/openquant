"""情绪增强事件数据源

在 CompositeEventSource 基础上，额外整合新闻情绪数据，
将新闻情绪转换为 EventFactor 格式，统一注入到情绪分析器中。

主要功能：
1. 获取历史事件因子（财报、分红、大宗交易等）
2. 获取新闻情绪并转换为 EventFactor
3. 将两类数据合并后提供给策略使用
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from openquant.core.models import (
    EventFactor,
    EventSentiment,
    EventType,
    MarketType,
)
from openquant.datasource.composite_event_source import CompositeEventSource
from openquant.datasource.news_sentiment_source import NewsSentimentSource

logger = logging.getLogger(__name__)


class SentimentEventSource:
    """情绪增强事件数据源

    整合历史事件因子和新闻情绪，统一以 EventFactor 列表返回。
    新闻情绪会被转换为 EventType.NEWS_POSITIVE / NEWS_NEGATIVE 类型的 EventFactor。
    """

    def __init__(
        self,
        enable_news: bool = True,
        enable_events: bool = True,
    ):
        self._enable_news = enable_news
        self._enable_events = enable_events
        self._event_source = CompositeEventSource()
        self._news_source = NewsSentimentSource()

    def get_name(self) -> str:
        return "sentiment_event"

    def fetch_all(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
        event_types: list[EventType] | None = None,
    ) -> list[EventFactor]:
        """获取所有情绪相关事件因子

        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            market: 市场类型
            event_types: 指定事件类型（None 则获取全部）

        Returns:
            合并后的 EventFactor 列表，按日期排序
        """
        all_events: list[EventFactor] = []

        # 获取历史事件因子
        if self._enable_events:
            try:
                events = self._event_source.fetch_events(
                    symbol, start_date, end_date, event_types, market,
                )
                all_events.extend(events)
                logger.info("获取 %s 历史事件因子 %d 条", symbol, len(events))
            except Exception as exc:
                logger.warning("获取 %s 历史事件因子失败: %s", symbol, exc)

        # 获取新闻情绪并转换为 EventFactor
        if self._enable_news:
            try:
                news_items = self._news_source.fetch_news_sentiment(
                    symbol, start_date, end_date,
                )
                news_events = self._convert_news_to_events(symbol, news_items, market)
                all_events.extend(news_events)
                logger.info("获取 %s 新闻情绪事件 %d 条", symbol, len(news_events))
            except Exception as exc:
                logger.warning("获取 %s 新闻情绪失败: %s", symbol, exc)

        all_events.sort(key=lambda e: e.event_date)
        return all_events

    def fetch_latest(
        self,
        symbol: str,
        market: MarketType = MarketType.A_SHARE,
        limit: int = 20,
    ) -> list[EventFactor]:
        """获取最新情绪事件（用于实时交易）"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        events = self.fetch_all(symbol, start_date, end_date, market)
        events.sort(key=lambda e: e.event_date, reverse=True)
        return events[:limit]

    def _convert_news_to_events(
        self,
        symbol: str,
        news_items: list[dict],
        market: MarketType,
    ) -> list[EventFactor]:
        """将新闻情绪数据转换为 EventFactor 格式"""
        events: list[EventFactor] = []

        for item in news_items:
            sentiment_label = item.get("sentiment", "neutral")
            score = float(item.get("score", 0.0))
            pub_time = item.get("datetime")
            title = item.get("title", "")

            if sentiment_label == "neutral" or abs(score) < 0.1:
                continue

            if sentiment_label == "positive":
                sentiment = EventSentiment.BULLISH
                event_type = EventType.NEWS_POSITIVE
                strength = min(abs(score), 2.0)
            else:
                sentiment = EventSentiment.BEARISH
                event_type = EventType.NEWS_NEGATIVE
                strength = min(abs(score), 2.0)

            try:
                event_date = pd.Timestamp(pub_time).to_pydatetime() if pub_time else datetime.now()
            except (ValueError, TypeError):
                continue

            events.append(EventFactor(
                symbol=symbol,
                event_date=event_date,
                event_type=event_type,
                sentiment=sentiment,
                strength=strength,
                description=title[:100] if title else "",
                source="news_sentiment",
                market=market,
                extra={"raw_score": score, "sentiment_label": sentiment_label},
            ))

        return events
