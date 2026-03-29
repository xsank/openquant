"""风控规则引擎

提供全局风控层，在订单执行前进行风控检查。
支持多种风控规则的组合使用。
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime

from openquant.core.models import Bar, Order, OrderSide, OrderStatus, Portfolio

logger = logging.getLogger(__name__)


class RiskRuleType(enum.Enum):
    """风控规则类型"""
    MAX_POSITION_PER_SYMBOL = "max_position_per_symbol"
    MAX_TOTAL_POSITION = "max_total_position"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses"
    MAX_SINGLE_ORDER_VALUE = "max_single_order_value"


@dataclass
class RiskRule:
    """风控规则配置

    Attributes:
        rule_type: 规则类型
        threshold: 阈值
        enabled: 是否启用
    """
    rule_type: RiskRuleType
    threshold: float
    enabled: bool = True


class RiskManager:
    """风控管理器

    在订单执行前进行风控检查，拦截不符合规则的订单。
    """

    def __init__(self, rules: list[RiskRule] | None = None):
        self._rules = rules or []
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = 0.0
        self._last_date: datetime | None = None
        self._consecutive_losses: int = 0
        self._is_suspended: bool = False

    @property
    def rules(self) -> list[RiskRule]:
        return self._rules

    @property
    def is_suspended(self) -> bool:
        return self._is_suspended

    def add_rule(self, rule: RiskRule) -> None:
        """添加风控规则"""
        self._rules.append(rule)

    def reset(self) -> None:
        """重置风控状态"""
        self._daily_pnl = 0.0
        self._daily_start_equity = 0.0
        self._last_date = None
        self._consecutive_losses = 0
        self._is_suspended = False

    def update_daily_state(self, current_datetime: datetime, portfolio: Portfolio) -> None:
        """更新每日状态，在每根 Bar 处理前调用"""
        current_date = current_datetime.date() if isinstance(current_datetime, datetime) else current_datetime

        if self._last_date is None or current_date != self._last_date:
            self._daily_start_equity = portfolio.total_equity
            self._daily_pnl = 0.0
            self._last_date = current_date
            # 新的一天解除暂停
            if self._is_suspended:
                logger.info("新交易日，解除风控暂停")
                self._is_suspended = False

        self._daily_pnl = portfolio.total_equity - self._daily_start_equity

    def on_trade_result(self, is_profitable: bool) -> None:
        """交易结果回调，用于跟踪连续亏损"""
        if is_profitable:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    def check_order(self, order: Order, bar: Bar, portfolio: Portfolio) -> tuple[bool, str]:
        """检查订单是否通过风控

        Args:
            order: 待检查的订单
            bar: 当前 K 线
            portfolio: 当前组合

        Returns:
            (是否通过, 拒绝原因)
        """
        if self._is_suspended:
            return False, "风控暂停交易中"

        # 卖出订单不做风控限制（允许止损卖出）
        if order.side == OrderSide.SELL:
            return True, ""

        for rule in self._rules:
            if not rule.enabled:
                continue

            passed, reason = self._check_single_rule(rule, order, bar, portfolio)
            if not passed:
                return False, reason

        return True, ""

    def _check_single_rule(
        self,
        rule: RiskRule,
        order: Order,
        bar: Bar,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """检查单条风控规则"""

        if rule.rule_type == RiskRuleType.MAX_POSITION_PER_SYMBOL:
            return self._check_max_position_per_symbol(rule, order, portfolio)

        elif rule.rule_type == RiskRuleType.MAX_TOTAL_POSITION:
            return self._check_max_total_position(rule, order, bar, portfolio)

        elif rule.rule_type == RiskRuleType.MAX_DAILY_LOSS:
            return self._check_max_daily_loss(rule, portfolio)

        elif rule.rule_type == RiskRuleType.MAX_CONSECUTIVE_LOSSES:
            return self._check_max_consecutive_losses(rule)

        elif rule.rule_type == RiskRuleType.MAX_SINGLE_ORDER_VALUE:
            return self._check_max_single_order_value(rule, order, portfolio)

        return True, ""

    def _check_max_position_per_symbol(
        self,
        rule: RiskRule,
        order: Order,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """单标的最大仓位占比检查"""
        if order.symbol in portfolio.positions:
            current_value = portfolio.positions[order.symbol].market_value
        else:
            current_value = 0.0

        new_value = current_value + order.price * order.quantity
        position_ratio = new_value / portfolio.total_equity if portfolio.total_equity > 0 else 0

        if position_ratio > rule.threshold:
            reason = (
                f"单标的仓位超限: {order.symbol} 仓位占比 {position_ratio:.1%} > {rule.threshold:.1%}"
            )
            logger.warning(reason)
            return False, reason

        return True, ""

    def _check_max_total_position(
        self,
        rule: RiskRule,
        order: Order,
        bar: Bar,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """总仓位占比检查"""
        new_total_market_value = portfolio.total_market_value + order.price * order.quantity
        total_ratio = new_total_market_value / portfolio.total_equity if portfolio.total_equity > 0 else 0

        if total_ratio > rule.threshold:
            reason = f"总仓位超限: {total_ratio:.1%} > {rule.threshold:.1%}"
            logger.warning(reason)
            return False, reason

        return True, ""

    def _check_max_daily_loss(
        self,
        rule: RiskRule,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """单日最大亏损检查"""
        if self._daily_start_equity <= 0:
            return True, ""

        daily_loss_pct = -self._daily_pnl / self._daily_start_equity
        if daily_loss_pct >= rule.threshold:
            self._is_suspended = True
            reason = f"单日亏损超限: {daily_loss_pct:.2%} >= {rule.threshold:.2%}，暂停交易"
            logger.warning(reason)
            return False, reason

        return True, ""

    def _check_max_consecutive_losses(self, rule: RiskRule) -> tuple[bool, str]:
        """连续亏损次数检查"""
        if self._consecutive_losses >= int(rule.threshold):
            self._is_suspended = True
            reason = (
                f"连续亏损超限: {self._consecutive_losses} 次 >= {int(rule.threshold)} 次，暂停交易"
            )
            logger.warning(reason)
            return False, reason

        return True, ""

    def _check_max_single_order_value(
        self,
        rule: RiskRule,
        order: Order,
        portfolio: Portfolio,
    ) -> tuple[bool, str]:
        """单笔订单金额占比检查"""
        order_value = order.price * order.quantity
        order_ratio = order_value / portfolio.total_equity if portfolio.total_equity > 0 else 0

        if order_ratio > rule.threshold:
            reason = f"单笔订单金额超限: {order_ratio:.1%} > {rule.threshold:.1%}"
            logger.warning(reason)
            return False, reason

        return True, ""
