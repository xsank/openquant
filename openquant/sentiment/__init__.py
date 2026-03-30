"""情绪分析模块

提供基于历史事件和实时新闻的情绪分析功能，
通过 SentimentStrategyWrapper 以装饰器模式零侵入地增强任意现有策略。

核心设计：
    SentimentStrategyWrapper.wrap(strategy, config) 可包装任意 StrategyInterface，
    通过 SentimentConfig.enabled 控制情绪功能的总开关。

快速使用：
    from openquant.sentiment import SentimentStrategyWrapper, SentimentConfig
    from openquant.strategy.macd_strategy import MACDStrategy

    strategy = SentimentStrategyWrapper.wrap(MACDStrategy(), SentimentConfig())
"""
from openquant.sentiment.config import SentimentConfig
from openquant.sentiment.sentiment_analyzer import SentimentAnalyzer, SentimentScore
from openquant.sentiment.strategy_wrapper import SentimentStrategyWrapper

__all__ = ["SentimentConfig", "SentimentAnalyzer", "SentimentScore", "SentimentStrategyWrapper"]
