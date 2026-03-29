"""止损止盈模块单元测试"""
import pytest
from datetime import datetime

from openquant.core.models import (
    Bar, MarketType, Order, OrderSide, Portfolio, Position,
)
from openquant.risk.stop_loss import StopLossConfig, StopLossManager, StopLossType


def _make_bar(symbol: str, close: float, dt: datetime | None = None) -> Bar:
    dt = dt or datetime(2025, 1, 1)
    return Bar(
        symbol=symbol, datetime=dt,
        open=close, high=close + 1, low=close - 1, close=close,
        volume=10000,
    )


def _make_sell_order(symbol, price, quantity, market=MarketType.A_SHARE):
    return Order(
        order_id="stop", symbol=symbol, side=OrderSide.SELL,
        price=price, quantity=quantity, created_at=datetime.now(),
        market=market,
    )


class TestStopLossDisabled:
    def test_disabled_returns_no_orders(self):
        config = StopLossConfig(enabled=False)
        manager = StopLossManager(config)
        portfolio = Portfolio(initial_capital=100000, cash=50000)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=8.0,
        )
        bar = _make_bar("600000", 8.0)
        orders = manager.check_stop(bar, portfolio, _make_sell_order)
        assert orders == []


class TestFixedPercentStopLoss:
    def test_triggers_stop_loss(self):
        config = StopLossConfig(
            stop_loss_type=StopLossType.FIXED_PERCENT,
            stop_loss_pct=0.05,
            take_profit_pct=None,
        )
        manager = StopLossManager(config)
        manager.on_order_filled("600000", OrderSide.BUY, 10.0)

        portfolio = Portfolio(initial_capital=100000, cash=50000)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=9.4,
        )
        bar = _make_bar("600000", 9.4)
        orders = manager.check_stop(bar, portfolio, _make_sell_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == 1000

    def test_no_trigger_within_threshold(self):
        config = StopLossConfig(
            stop_loss_type=StopLossType.FIXED_PERCENT,
            stop_loss_pct=0.05,
            take_profit_pct=None,
        )
        manager = StopLossManager(config)
        manager.on_order_filled("600000", OrderSide.BUY, 10.0)

        portfolio = Portfolio(initial_capital=100000, cash=50000)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=9.6,
        )
        bar = _make_bar("600000", 9.6)
        orders = manager.check_stop(bar, portfolio, _make_sell_order)
        assert orders == []


class TestTakeProfit:
    def test_triggers_take_profit(self):
        config = StopLossConfig(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        manager = StopLossManager(config)
        manager.on_order_filled("600000", OrderSide.BUY, 10.0)

        portfolio = Portfolio(initial_capital=100000, cash=50000)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=11.1,
        )
        bar = _make_bar("600000", 11.1)
        orders = manager.check_stop(bar, portfolio, _make_sell_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL


class TestTrailingStop:
    def test_triggers_trailing_stop(self):
        config = StopLossConfig(
            stop_loss_type=StopLossType.TRAILING_PERCENT,
            trailing_pct=0.05,
            take_profit_pct=None,
        )
        manager = StopLossManager(config)
        manager.on_order_filled("600000", OrderSide.BUY, 10.0)

        portfolio = Portfolio(initial_capital=100000, cash=50000)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=12.0,
        )

        bar1 = _make_bar("600000", 12.0)
        orders = manager.check_stop(bar1, portfolio, _make_sell_order)
        assert orders == []

        portfolio.positions["600000"].current_price = 11.3
        bar2 = _make_bar("600000", 11.3)
        orders = manager.check_stop(bar2, portfolio, _make_sell_order)
        assert len(orders) == 1


class TestATRStop:
    def test_triggers_atr_stop(self):
        config = StopLossConfig(
            stop_loss_type=StopLossType.ATR_BASED,
            atr_multiplier=2.0,
            take_profit_pct=None,
        )
        manager = StopLossManager(config)
        manager.on_order_filled("600000", OrderSide.BUY, 10.0)
        manager.update_atr("600000", 0.5)

        portfolio = Portfolio(initial_capital=100000, cash=50000)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=8.9,
        )
        bar = _make_bar("600000", 8.9)
        orders = manager.check_stop(bar, portfolio, _make_sell_order)
        assert len(orders) == 1

    def test_no_trigger_above_stop_price(self):
        config = StopLossConfig(
            stop_loss_type=StopLossType.ATR_BASED,
            atr_multiplier=2.0,
            take_profit_pct=None,
        )
        manager = StopLossManager(config)
        manager.on_order_filled("600000", OrderSide.BUY, 10.0)
        manager.update_atr("600000", 0.5)

        portfolio = Portfolio(initial_capital=100000, cash=50000)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=9.1,
        )
        bar = _make_bar("600000", 9.1)
        orders = manager.check_stop(bar, portfolio, _make_sell_order)
        assert orders == []


class TestStopLossManagerReset:
    def test_reset_clears_state(self):
        manager = StopLossManager(StopLossConfig())
        manager.on_order_filled("600000", OrderSide.BUY, 10.0)
        manager.update_atr("600000", 0.5)
        manager.reset()
        assert manager._highest_prices == {}
        assert manager._entry_prices == {}
        assert manager._atr_values == {}


class TestNoPosition:
    def test_no_position_returns_empty(self):
        manager = StopLossManager(StopLossConfig())
        portfolio = Portfolio(initial_capital=100000, cash=100000)
        bar = _make_bar("600000", 10.0)
        orders = manager.check_stop(bar, portfolio, _make_sell_order)
        assert orders == []
