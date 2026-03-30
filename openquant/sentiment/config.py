"""情绪分析模块配置

通过 SentimentConfig 控制情绪分析功能的开关和参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SentimentConfig:
    """情绪分析配置

    通过 enabled 字段控制是否开启情绪分析功能。
    关闭时，情绪权重因子不生效，策略退化为纯技术面策略。

    Attributes:
        enabled: 是否开启情绪分析（总开关）
        use_news_sentiment: 是否使用实时新闻情绪（需要网络请求）
        use_event_sentiment: 是否使用历史事件情绪（基于 EventFactor）
        lookback_days: 情绪回看天数
        decay_factor: 情绪时间衰减系数（越小衰减越快）
        bullish_weight_boost: 正向情绪对买入信号的权重加成（0~1）
        bearish_weight_penalty: 负向情绪对买入信号的权重惩罚（0~1）
        bullish_sell_delay_threshold: 正向情绪延迟卖出的得分阈值
        bearish_force_sell_threshold: 负向情绪强制卖出的得分阈值（负数）
        bullish_block_buy_threshold: 正向情绪触发加仓的得分阈值
        bearish_block_buy_threshold: 负向情绪阻止买入的得分阈值（负数）
        news_max_age_hours: 新闻情绪的最大有效时长（小时）
        sentiment_score_clip: 情绪得分截断范围（防止极端值）
        position_boost_ratio: 正向情绪时额外加仓比例
        position_penalty_ratio: 负向情绪时减少仓位比例
    """
    enabled: bool = True
    use_news_sentiment: bool = True
    use_event_sentiment: bool = True
    lookback_days: int = 7
    decay_factor: float = 0.85
    bullish_weight_boost: float = 0.20
    bearish_weight_penalty: float = 0.20
    bullish_sell_delay_threshold: float = 1.0
    bearish_force_sell_threshold: float = -1.5
    bullish_block_buy_threshold: float = 0.5
    bearish_block_buy_threshold: float = -0.5
    news_max_age_hours: int = 48
    sentiment_score_clip: float = 3.0
    position_boost_ratio: float = 0.15
    position_penalty_ratio: float = 0.15

    @classmethod
    def disabled(cls) -> "SentimentConfig":
        """创建一个关闭情绪分析的配置"""
        return cls(enabled=False)

    @classmethod
    def conservative(cls) -> "SentimentConfig":
        """保守配置：情绪影响较小"""
        return cls(
            enabled=True,
            bullish_weight_boost=0.10,
            bearish_weight_penalty=0.10,
            position_boost_ratio=0.08,
            position_penalty_ratio=0.08,
            bearish_force_sell_threshold=-2.0,
            bullish_sell_delay_threshold=1.5,
        )

    @classmethod
    def aggressive(cls) -> "SentimentConfig":
        """激进配置：情绪影响较大"""
        return cls(
            enabled=True,
            bullish_weight_boost=0.30,
            bearish_weight_penalty=0.30,
            position_boost_ratio=0.25,
            position_penalty_ratio=0.25,
            bearish_force_sell_threshold=-1.0,
            bullish_sell_delay_threshold=0.8,
        )
