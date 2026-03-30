"""情绪策略包装器

通过装饰器模式将情绪分析能力注入到任意现有策略，无需修改原策略代码。

设计思路：
    SentimentStrategyWrapper 实现 StrategyInterface，
    内部持有一个被包装的原始策略（inner_strategy）。
    在每个 on_bar 调用中：
      1. 先计算当前情绪得分
      2. 将情绪得分注入 bar 的上下文（通过 _SentimentContext）
      3. 调用原始策略的 on_bar 获取原始订单列表
      4. 根据情绪得分对订单进行修正（调整数量、过滤、追加强制卖出）

使用示例：
    from openquant.sentiment.strategy_wrapper import SentimentStrategyWrapper
    from openquant.sentiment.config import SentimentConfig
    from openquant.strategy.macd_strategy import MACDStrategy

    # 方式一：直接包装
    strategy = SentimentStrategyWrapper(
        inner_strategy=MACDStrategy(),
        sentiment_config=SentimentConfig(),
    )

    # 方式二：关闭情绪（退化为原始策略）
    strategy = SentimentStrategyWrapper(
        inner_strategy=MACDStrategy(),
        sentiment_config=SentimentConfig.disabled(),
    )

    # 方式三：使用工厂方法
    strategy = SentimentStrategyWrapper.wrap(MACDStrategy())
"""
from __future__ import annotations

import logging
from datetime import datetime

from openquant.core.interfaces import StrategyInterface
from openquant.core.models import (
    Bar,
    EventFactor,
    Order,
    OrderSide,
    Portfolio,
)
from openquant.sentiment.config import SentimentConfig
from openquant.sentiment.sentiment_analyzer import SentimentAnalyzer, SentimentScore

logger = logging.getLogger(__name__)


class SentimentStrategyWrapper(StrategyInterface):
    """情绪策略包装器

    将情绪分析能力以零侵入的方式叠加到任意现有策略上。
    通过 SentimentConfig.enabled 控制情绪功能的总开关。

    情绪对订单的影响：
    - 买入订单：根据情绪净权重（net_weight）等比调整买入数量
      - 正向情绪 → net_weight > 1 → 数量增加（上限 lot_size 取整）
      - 负向情绪 → net_weight < 1 → 数量减少
      - 极端负向情绪 → 直接过滤掉买入订单（阻止买入）
    - 卖出订单：
      - 极端负向情绪 → 保留并提前执行
      - 强正向情绪 → 过滤掉卖出订单（延迟卖出）
    - 强制卖出：持仓期间若情绪极端负向，主动追加卖出订单
    """

    def __init__(
        self,
        inner_strategy: StrategyInterface,
        sentiment_config: SentimentConfig | None = None,
        lot_size: int = 100,
    ):
        """
        Args:
            inner_strategy: 被包装的原始策略实例
            sentiment_config: 情绪分析配置，None 则使用默认配置
            lot_size: 最小交易单位（手），用于调整数量时取整
        """
        self._inner = inner_strategy
        self._config = sentiment_config or SentimentConfig()
        self._analyzer = SentimentAnalyzer(self._config)
        self._lot_size = lot_size

    # ------------------------------------------------------------------ #
    # 工厂方法
    # ------------------------------------------------------------------ #

    @classmethod
    def wrap(
        cls,
        strategy: StrategyInterface,
        sentiment_config: SentimentConfig | None = None,
    ) -> "SentimentStrategyWrapper":
        """快捷包装方法

        Args:
            strategy: 任意现有策略实例
            sentiment_config: 情绪配置，None 则使用默认配置

        Returns:
            包装后的情绪增强策略
        """
        return cls(inner_strategy=strategy, sentiment_config=sentiment_config)

    # ------------------------------------------------------------------ #
    # StrategyInterface 实现
    # ------------------------------------------------------------------ #

    def get_name(self) -> str:
        base_name = self._inner.get_name()
        if not self._config.enabled:
            return base_name
        return f"{base_name}+Sentiment"

    def initialize(self, portfolio: Portfolio) -> None:
        self._inner.initialize(portfolio)

    def on_order_filled(self, order: Order, portfolio: Portfolio) -> None:
        self._inner.on_order_filled(order, portfolio)

    def on_finish(self, portfolio: Portfolio) -> None:
        self._inner.on_finish(portfolio)

    def load_events(self, symbol: str, events: list[EventFactor]) -> None:
        """加载事件因子（同时注入原策略和情绪分析器）"""
        if hasattr(self._inner, "load_events"):
            self._inner.load_events(symbol, events)
        self._analyzer.load_events(symbol, events)

    def load_news(self, symbol: str, news_items: list[dict]) -> None:
        """加载新闻情绪数据到情绪分析器"""
        self._analyzer.load_news(symbol, news_items)

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        """核心逻辑：情绪感知的 on_bar

        流程：
        1. 计算当前情绪得分
        2. 若情绪关闭，直接透传原策略结果
        3. 检查是否需要强制卖出（不依赖原策略信号）
        4. 调用原策略获取原始订单
        5. 对原始订单逐一应用情绪修正
        """
        # 情绪关闭时完全透传，零开销
        if not self._config.enabled:
            return self._inner.on_bar(bar, portfolio)

        sentiment_score = self._analyzer.compute_sentiment(bar.symbol, bar.datetime)

        if sentiment_score.event_count > 0 or sentiment_score.news_count > 0:
            logger.debug(
                "[%s] %s 情绪: %s",
                bar.datetime.strftime("%Y-%m-%d"), bar.symbol, sentiment_score,
            )

        # 检查是否需要强制卖出（不等原策略信号）
        force_sell_orders = self._maybe_force_sell(bar, portfolio, sentiment_score)
        if force_sell_orders:
            return force_sell_orders

        # 获取原策略的原始订单
        raw_orders = self._inner.on_bar(bar, portfolio)

        # 对每个订单应用情绪修正
        adjusted_orders: list[Order] = []
        for order in raw_orders:
            modified = self._apply_sentiment_to_order(order, bar, portfolio, sentiment_score)
            if modified is not None:
                adjusted_orders.append(modified)

        return adjusted_orders

    # ------------------------------------------------------------------ #
    # 情绪修正逻辑
    # ------------------------------------------------------------------ #

    def _maybe_force_sell(
        self,
        bar: Bar,
        portfolio: Portfolio,
        score: SentimentScore,
    ) -> list[Order]:
        """极端负向情绪时主动追加强制卖出订单"""
        if not self._analyzer.should_force_sell(score):
            return []

        position = portfolio.positions.get(bar.symbol)
        if position is None or position.quantity <= 0:
            return []

        logger.info(
            "[%s] %s 情绪强制卖出: composite=%.3f (阈值=%.2f)",
            bar.datetime.strftime("%Y-%m-%d"), bar.symbol,
            score.composite_score, self._config.bearish_force_sell_threshold,
        )

        # 复用原策略的 create_sell_order（如果有），否则手动构建
        if hasattr(self._inner, "create_sell_order"):
            return [self._inner.create_sell_order(bar.symbol, bar.close, position.quantity, bar.market)]

        import uuid
        from openquant.core.models import OrderSide, OrderStatus
        return [Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=bar.symbol,
            side=OrderSide.SELL,
            price=bar.close,
            quantity=position.quantity,
            created_at=bar.datetime,
            market=bar.market,
        )]

    def _apply_sentiment_to_order(
        self,
        order: Order,
        bar: Bar,
        portfolio: Portfolio,
        score: SentimentScore,
    ) -> Order | None:
        """对单个订单应用情绪修正

        Returns:
            修正后的订单，None 表示该订单被情绪过滤掉
        """
        if order.side == OrderSide.BUY:
            return self._adjust_buy_order(order, bar, score)
        else:
            return self._adjust_sell_order(order, bar, score)

    def _adjust_buy_order(
        self,
        order: Order,
        bar: Bar,
        score: SentimentScore,
    ) -> Order | None:
        """调整买入订单

        - 极端负向情绪 → 过滤（返回 None）
        - 其他情况 → 按 net_weight 等比调整数量
        """
        if self._analyzer.should_block_buy(score):
            logger.info(
                "[%s] %s 情绪阻止买入: composite=%.3f (阈值=%.2f)",
                bar.datetime.strftime("%Y-%m-%d"), bar.symbol,
                score.composite_score, self._config.bearish_block_buy_threshold,
            )
            return None

        net_weight = score.net_weight
        if abs(net_weight - 1.0) < 0.01:
            return order

        # 按权重调整数量，保持手数取整
        raw_quantity = order.quantity * net_weight
        adjusted_quantity = max(
            self._lot_size,
            int(raw_quantity / self._lot_size) * self._lot_size,
        )

        if adjusted_quantity != order.quantity:
            logger.info(
                "[%s] %s 情绪调整买入数量: %d → %d (net_weight=%.3f, composite=%.3f)",
                bar.datetime.strftime("%Y-%m-%d"), bar.symbol,
                order.quantity, adjusted_quantity,
                net_weight, score.composite_score,
            )
            order.quantity = adjusted_quantity

        return order

    def _adjust_sell_order(
        self,
        order: Order,
        bar: Bar,
        score: SentimentScore,
    ) -> Order | None:
        """调整卖出订单

        - 强正向情绪 → 延迟卖出（过滤，返回 None）
        - 其他情况 → 保留原卖出订单
        """
        if self._analyzer.should_delay_sell(score):
            logger.info(
                "[%s] %s 情绪延迟卖出: composite=%.3f (阈值=%.2f)",
                bar.datetime.strftime("%Y-%m-%d"), bar.symbol,
                score.composite_score, self._config.bullish_sell_delay_threshold,
            )
            return None

        return order

    # ------------------------------------------------------------------ #
    # 属性访问
    # ------------------------------------------------------------------ #

    @property
    def inner_strategy(self) -> StrategyInterface:
        """获取被包装的原始策略"""
        return self._inner

    @property
    def sentiment_config(self) -> SentimentConfig:
        """获取情绪配置"""
        return self._config

    @property
    def sentiment_analyzer(self) -> SentimentAnalyzer:
        """获取情绪分析器"""
        return self._analyzer
