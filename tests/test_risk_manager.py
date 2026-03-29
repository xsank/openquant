"""风控规则引擎单元测试"""
import pytest
from datetime import datetime

from openquant.core.models import (
    Bar, MarketType, Order, OrderSide, OrderStatus, Portfolio, Position,
)
from openquant.risk.risk_manager import RiskManager, RiskRule, RiskRuleType


def _make_bar(symbol: str, close: float, dt: datetime | None = None) -> Bar:
    dt = dt or datetime(2025, 1, 1)
    return Bar(
        symbol=symbol, datetime=dt,
        open=close, high=close + 1, low=close - 1, close=close,
        volume=10000,
    )


def _make_buy_order(symbol: str, price: float, quantity: int) -> Order:
    return Order(
        order_id="test", symbol=symbol, side=OrderSide.BUY,
        price=price, quantity=quantity, created_at=datetime.now(),
    )


def _make_sell_order(symbol: str, price: float, quantity: int) -> Order:
    return Order(
        order_id="test", symbol=symbol, side=OrderSide.SELL,
        price=price, quantity=quantity, created_at=datetime.now(),
    )


class TestSellOrderBypass:
    def test_sell_order_always_passes(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_TOTAL_POSITION, threshold=0.0),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=50000)
        bar = _make_bar("600000", 10.0)
        order = _make_sell_order("600000", 10.0, 100)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is True


class TestMaxPositionPerSymbol:
    def test_blocks_over_limit(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_POSITION_PER_SYMBOL, threshold=0.3),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 4000)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is False
        assert "单标的仓位超限" in reason

    def test_allows_within_limit(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_POSITION_PER_SYMBOL, threshold=0.3),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 2000)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is True


class TestMaxTotalPosition:
    def test_blocks_over_total_limit(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_TOTAL_POSITION, threshold=0.8),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=20000)
        portfolio.positions["00700"] = Position(
            symbol="00700", quantity=200, avg_cost=300.0, current_price=350.0,
        )
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 2000)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is False
        assert "总仓位超限" in reason


class TestMaxDailyLoss:
    def test_blocks_after_daily_loss(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_DAILY_LOSS, threshold=0.03),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=96000)
        bar = _make_bar("600000", 10.0, datetime(2025, 1, 1, 10, 0))

        manager.update_daily_state(
            datetime(2025, 1, 1, 9, 30),
            Portfolio(initial_capital=100000, cash=100000),
        )
        manager.update_daily_state(datetime(2025, 1, 1, 10, 0), portfolio)

        order = _make_buy_order("600000", 10.0, 100)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is False
        assert "单日亏损超限" in reason
        assert manager.is_suspended is True


class TestMaxConsecutiveLosses:
    def test_blocks_after_consecutive_losses(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_CONSECUTIVE_LOSSES, threshold=3),
        ])
        manager.on_trade_result(False)
        manager.on_trade_result(False)
        manager.on_trade_result(False)

        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 100)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is False
        assert "连续亏损超限" in reason

    def test_resets_on_win(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_CONSECUTIVE_LOSSES, threshold=3),
        ])
        manager.on_trade_result(False)
        manager.on_trade_result(False)
        manager.on_trade_result(True)
        manager.on_trade_result(False)

        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 100)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is True


class TestMaxSingleOrderValue:
    def test_blocks_large_order(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_SINGLE_ORDER_VALUE, threshold=0.2),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 3000)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is False
        assert "单笔订单金额超限" in reason


class TestMultipleRules:
    def test_all_rules_must_pass(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_POSITION_PER_SYMBOL, threshold=0.5),
            RiskRule(RiskRuleType.MAX_SINGLE_ORDER_VALUE, threshold=0.1),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 2000)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is False


class TestDisabledRule:
    def test_disabled_rule_skipped(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_SINGLE_ORDER_VALUE, threshold=0.01, enabled=False),
        ])
        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 5000)
        passed, reason = manager.check_order(order, bar, portfolio)
        assert passed is True


class TestSuspendedState:
    def test_suspended_blocks_all(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_CONSECUTIVE_LOSSES, threshold=2),
        ])
        manager.on_trade_result(False)
        manager.on_trade_result(False)

        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 100)
        passed1, _ = manager.check_order(order, bar, portfolio)
        assert passed1 is False
        passed2, reason = manager.check_order(order, bar, portfolio)
        assert passed2 is False
        assert "风控暂停交易中" in reason

    def test_new_day_lifts_suspension(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_CONSECUTIVE_LOSSES, threshold=2),
        ])
        manager.on_trade_result(False)
        manager.on_trade_result(False)

        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        order = _make_buy_order("600000", 10.0, 100)
        manager.check_order(order, bar, portfolio)

        manager.update_daily_state(datetime(2025, 1, 2), portfolio)
        assert manager.is_suspended is False


class TestReset:
    def test_reset_clears_all(self):
        manager = RiskManager([
            RiskRule(RiskRuleType.MAX_CONSECUTIVE_LOSSES, threshold=2),
        ])
        manager.on_trade_result(False)
        manager.on_trade_result(False)
        manager.reset()
        assert manager._consecutive_losses == 0
        assert manager._is_suspended is False
