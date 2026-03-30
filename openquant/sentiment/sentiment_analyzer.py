"""情绪分析器

整合历史事件情绪和实时新闻情绪，计算综合情绪得分。
支持时间衰减加权、正负向情绪分离，以及权重因子输出。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from openquant.core.models import EventFactor, EventSentiment, MarketType
from openquant.sentiment.config import SentimentConfig

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """情绪得分结果

    Attributes:
        composite_score: 综合情绪得分（正为利多，负为利空）
        bullish_score: 正向情绪得分（>=0）
        bearish_score: 负向情绪得分（<=0）
        bullish_weight: 正向情绪权重因子（>=1.0，用于放大买入信号）
        bearish_weight: 负向情绪权重因子（<=1.0，用于抑制买入信号）
        event_count: 参与计算的事件数量
        news_count: 参与计算的新闻数量
        is_sentiment_enabled: 情绪分析是否已开启
    """
    composite_score: float = 0.0
    bullish_score: float = 0.0
    bearish_score: float = 0.0
    bullish_weight: float = 1.0
    bearish_weight: float = 1.0
    event_count: int = 0
    news_count: int = 0
    is_sentiment_enabled: bool = True

    @property
    def net_weight(self) -> float:
        """综合权重因子：正向情绪提升，负向情绪压制"""
        return self.bullish_weight * self.bearish_weight

    @property
    def is_bullish(self) -> bool:
        return self.composite_score > 0

    @property
    def is_bearish(self) -> bool:
        return self.composite_score < 0

    def __repr__(self) -> str:
        return (
            f"SentimentScore(composite={self.composite_score:.3f}, "
            f"bullish={self.bullish_score:.3f}, bearish={self.bearish_score:.3f}, "
            f"net_weight={self.net_weight:.3f}, "
            f"events={self.event_count}, news={self.news_count})"
        )


class SentimentAnalyzer:
    """情绪分析器

    整合历史事件情绪（EventFactor）和实时新闻情绪，
    计算综合情绪得分和权重因子，供策略决策使用。

    使用方式：
        config = SentimentConfig(enabled=True)
        analyzer = SentimentAnalyzer(config)
        analyzer.load_events(symbol, events)
        score = analyzer.compute_sentiment(symbol, current_date)
        # 根据 score.bullish_weight / bearish_weight 调整仓位
    """

    def __init__(self, config: SentimentConfig | None = None):
        self._config = config or SentimentConfig()
        self._event_store: dict[str, list[EventFactor]] = {}
        self._news_store: dict[str, list[dict]] = {}

    @property
    def config(self) -> SentimentConfig:
        return self._config

    def load_events(self, symbol: str, events: list[EventFactor]) -> None:
        """加载历史事件因子数据"""
        self._event_store[symbol] = sorted(events, key=lambda e: e.event_date)
        logger.debug("情绪分析器加载 %s 的 %d 个事件因子", symbol, len(events))

    def load_news(self, symbol: str, news_items: list[dict]) -> None:
        """加载新闻情绪数据

        Args:
            symbol: 标的代码
            news_items: 新闻列表，每条包含 datetime, sentiment, score, title 字段
        """
        self._news_store[symbol] = sorted(
            news_items, key=lambda n: n.get("datetime", datetime.min),
        )
        logger.debug("情绪分析器加载 %s 的 %d 条新闻", symbol, len(news_items))

    def compute_sentiment(
        self,
        symbol: str,
        current_date: datetime,
        lookback_days: int | None = None,
    ) -> SentimentScore:
        """计算综合情绪得分

        Args:
            symbol: 标的代码
            current_date: 当前日期
            lookback_days: 回看天数（None 则使用配置值）

        Returns:
            SentimentScore 对象，包含综合得分和权重因子
        """
        if not self._config.enabled:
            return SentimentScore(is_sentiment_enabled=False)

        effective_lookback = lookback_days or self._config.lookback_days

        event_bullish = 0.0
        event_bearish = 0.0
        event_count = 0

        if self._config.use_event_sentiment:
            event_bullish, event_bearish, event_count = self._compute_event_sentiment(
                symbol, current_date, effective_lookback,
            )

        news_bullish = 0.0
        news_bearish = 0.0
        news_count = 0

        if self._config.use_news_sentiment:
            news_bullish, news_bearish, news_count = self._compute_news_sentiment(
                symbol, current_date, effective_lookback,
            )

        total_bullish = event_bullish + news_bullish
        total_bearish = event_bearish + news_bearish

        # 截断极端值
        clip = self._config.sentiment_score_clip
        total_bullish = min(total_bullish, clip)
        total_bearish = max(total_bearish, -clip)

        composite_score = total_bullish + total_bearish

        # 计算权重因子
        bullish_weight, bearish_weight = self._compute_weight_factors(
            total_bullish, total_bearish,
        )

        return SentimentScore(
            composite_score=composite_score,
            bullish_score=total_bullish,
            bearish_score=total_bearish,
            bullish_weight=bullish_weight,
            bearish_weight=bearish_weight,
            event_count=event_count,
            news_count=news_count,
            is_sentiment_enabled=True,
        )

    def _compute_event_sentiment(
        self,
        symbol: str,
        current_date: datetime,
        lookback_days: int,
    ) -> tuple[float, float, int]:
        """计算历史事件情绪得分

        Returns:
            (bullish_score, bearish_score, event_count)
        """
        events = self._event_store.get(symbol, [])
        if not events:
            return 0.0, 0.0, 0

        current_ts = pd.Timestamp(current_date)
        start_ts = current_ts - pd.Timedelta(days=lookback_days)

        recent_events = [
            e for e in events
            if start_ts <= pd.Timestamp(e.event_date) <= current_ts
        ]

        if not recent_events:
            return 0.0, 0.0, 0

        bullish_score = 0.0
        bearish_score = 0.0

        for event in recent_events:
            days_ago = (current_ts - pd.Timestamp(event.event_date)).days
            time_weight = self._config.decay_factor ** max(days_ago, 0)

            if event.sentiment == EventSentiment.BULLISH:
                bullish_score += event.strength * time_weight
            elif event.sentiment == EventSentiment.BEARISH:
                bearish_score -= event.strength * time_weight

        return bullish_score, bearish_score, len(recent_events)

    def _compute_news_sentiment(
        self,
        symbol: str,
        current_date: datetime,
        lookback_days: int,
    ) -> tuple[float, float, int]:
        """计算新闻情绪得分

        Returns:
            (bullish_score, bearish_score, news_count)
        """
        news_items = self._news_store.get(symbol, [])
        if not news_items:
            return 0.0, 0.0, 0

        current_ts = pd.Timestamp(current_date)
        start_ts = current_ts - pd.Timedelta(days=lookback_days)

        # 软过滤：接受 start_ts 之后的所有新闻（包括晚于 current_date 的最新新闻）
        # 晚于 current_date 的新闻 days_ago 为负，统一按 0 处理（最大权重），
        # 因为这类新闻是接口能获取到的最新数据，代表当前市场情绪。
        recent_news = [
            item for item in news_items
            if pd.Timestamp(item.get("datetime", datetime.min)) >= start_ts
        ]

        if not recent_news:
            return 0.0, 0.0, 0

        bullish_score = 0.0
        bearish_score = 0.0

        for item in recent_news:
            news_dt = pd.Timestamp(item.get("datetime", current_date))
            # 未来新闻（晚于 current_date）按 days_ago=0 处理，给予最大权重
            days_ago = max((current_ts - news_dt).days, 0)
            time_weight = self._config.decay_factor ** days_ago

            raw_score = float(item.get("score", 0.0))
            sentiment_label = item.get("sentiment", "neutral")

            if sentiment_label == "positive" or raw_score > 0:
                bullish_score += abs(raw_score) * time_weight
            elif sentiment_label == "negative" or raw_score < 0:
                bearish_score -= abs(raw_score) * time_weight

        return bullish_score, bearish_score, len(recent_news)

    def _compute_weight_factors(
        self,
        bullish_score: float,
        bearish_score: float,
    ) -> tuple[float, float]:
        """将情绪得分转换为权重因子

        正向情绪 → bullish_weight > 1.0（放大买入信号）
        负向情绪 → bearish_weight < 1.0（抑制买入信号）

        Returns:
            (bullish_weight, bearish_weight)
        """
        clip = self._config.sentiment_score_clip

        # 正向权重：得分越高，权重越大，上限为 1 + bullish_weight_boost
        if bullish_score > 0:
            normalized_bullish = min(bullish_score / clip, 1.0)
            bullish_weight = 1.0 + normalized_bullish * self._config.bullish_weight_boost
        else:
            bullish_weight = 1.0

        # 负向权重：得分越低，权重越小，下限为 1 - bearish_weight_penalty
        if bearish_score < 0:
            normalized_bearish = min(abs(bearish_score) / clip, 1.0)
            bearish_weight = 1.0 - normalized_bearish * self._config.bearish_weight_penalty
        else:
            bearish_weight = 1.0

        return bullish_weight, bearish_weight

    def should_block_buy(self, score: SentimentScore) -> bool:
        """判断是否应阻止买入（负向情绪过强）"""
        if not self._config.enabled:
            return False
        return score.composite_score < self._config.bearish_block_buy_threshold

    def should_boost_buy(self, score: SentimentScore) -> bool:
        """判断是否应加仓买入（正向情绪较强）"""
        if not self._config.enabled:
            return False
        return score.composite_score > self._config.bullish_block_buy_threshold

    def should_force_sell(self, score: SentimentScore) -> bool:
        """判断是否应强制卖出（负向情绪极强）"""
        if not self._config.enabled:
            return False
        return score.composite_score < self._config.bearish_force_sell_threshold

    def should_delay_sell(self, score: SentimentScore) -> bool:
        """判断是否应延迟卖出（正向情绪较强）"""
        if not self._config.enabled:
            return False
        return score.composite_score > self._config.bullish_sell_delay_threshold
