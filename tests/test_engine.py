"""回测引擎和策略集成单元测试"""
import pytest
import pandas as pd
from datetime import datetime

from openquant.core.models import (
    Bar, MarketType, Order, OrderSide, OrderStatus, Portfolio, Position,
)
from openquant.engine.backtest_engine import BacktestEngine
from openquant.strategy.base import BaseStrategy
from openquant.strategy.ma_cross import MACrossStrategy
from openquant.risk.stop_loss import StopLossConfig, StopLossType
from openquant.risk.risk_manager import RiskManager, RiskRule, RiskRuleType

def _make_test_data(prices: list[float], symbol: str = "600000") -> pd.DataFrame:
    """构造测试用 K 线数据"""
    dates = pd.date_range("2025-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({
        "datetime": dates,
        "open": [p * 0.99 for p in prices],
        "high": [p * 1.02 for p in prices],
        "low": [p * 0.98 for p in prices],
        "close": prices,
        "volume": [1000000] * len(prices),
    })

class AlwaysBuyStrategy(BaseStrategy):
    """测试用策略：第一根 Bar 全仓买入"""
    def __init__(self):
        super().__init__()
        self._bought = False

    def get_name(self) -> str:
        return "AlwaysBuy"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        if not self._bought and bar.symbol not in portfolio.positions:
            quantity = self.calculate_max_buyable(bar.close, portfolio.cash * 0.9)
            if quantity > 0:
                self._bought = True
                return [self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)]
        return []

class BuyThenSellStrategy(BaseStrategy):
    """测试用策略：第 5 根 Bar 买入，第 10 根 Bar 卖出"""
    def __init__(self):
        super().__init__()

    def get_name(self) -> str:
        return "BuyThenSell"

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Order]:
        self._record_bar(bar)
        bar_count = self.get_bar_count(bar.symbol)
        has_position = bar.symbol in portfolio.positions and portfolio.positions[bar.symbol].quantity > 0

        if bar_count == 5 and not has_position:
            quantity = self.calculate_max_buyable(bar.close, portfolio.cash * 0.9)
            if quantity > 0:
                return [self.create_buy_order(bar.symbol, bar.close, quantity, bar.market)]
        elif bar_count == 10 and has_position:
            pos = portfolio.positions[bar.symbol]
            return [self.create_sell_order(bar.symbol, bar.close, pos.quantity, bar.market)]
        return []

class TestBacktestEngineBasic:
    def test_no_strategy_raises(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="未设置策略"):
            engine.run()

    def test_no_data_raises(self):
        engine = BacktestEngine()
        engine.set_strategy(MACrossStrategy())
        with pytest.raises(ValueError, match="未添加数据"):
            engine.run()

    def test_missing_columns_raises(self):
        engine = BacktestEngine()
        df = pd.DataFrame({"datetime": [datetime(2025, 1, 1)], "close": [10.0]})
        with pytest.raises(ValueError, match="数据缺少必要列"):
            engine.add_data("600000", df)

    def test_basic_run(self):
        prices = [10.0] * 30
        engine = BacktestEngine(initial_capital=100000)
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        portfolio = engine.run()
        assert portfolio is not None
        assert portfolio.initial_capital == 100000

    def test_buy_and_sell(self):
        prices = [10.0 + i * 0.1 for i in range(15)]
        engine = BacktestEngine(initial_capital=100000, commission_rate=0.0, slippage_rate=0.0)
        engine.set_strategy(BuyThenSellStrategy())
        engine.add_data("600000", _make_test_data(prices))
        portfolio = engine.run()
        # Should have 2 trades (1 buy + 1 sell)
        assert len(portfolio.trade_history) == 2
        assert portfolio.trade_history[0].side == OrderSide.BUY
        assert portfolio.trade_history[1].side == OrderSide.SELL
        # After selling, no positions
        assert len(portfolio.positions) == 0

    def test_commission_deducted(self):
        prices = [10.0] * 30
        engine = BacktestEngine(initial_capital=100000, commission_rate=0.001, slippage_rate=0.0)
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        portfolio = engine.run()
        assert portfolio.total_commission > 0
        # Equity should be less than initial due to commission
        assert portfolio.total_equity < 100000

    def test_slippage_applied(self):
        prices = [10.0] * 30
        engine = BacktestEngine(initial_capital=100000, commission_rate=0.0, slippage_rate=0.01)
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        portfolio = engine.run()
        # Buy with slippage means fill_price > order price
        buy_trade = portfolio.trade_history[0]
        assert buy_trade.price > 10.0

class TestBacktestEngineResults:
    def test_get_results(self):
        prices = [10.0 + i * 0.1 for i in range(30)]
        engine = BacktestEngine(initial_capital=100000)
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        engine.run()
        results = engine.get_results()
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert results["strategy_name"] == "AlwaysBuy"
        assert results["initial_capital"] == 100000

    def test_get_equity_curve(self):
        prices = [10.0] * 10
        engine = BacktestEngine(initial_capital=100000)
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        engine.run()
        curve = engine.get_equity_curve()
        assert isinstance(curve, pd.DataFrame)
        assert "datetime" in curve.columns
        assert "equity" in curve.columns
        assert len(curve) == 10

    def test_empty_results(self):
        engine = BacktestEngine()
        assert engine.get_results() == {}
        assert engine.get_equity_curve().empty

class TestBacktestWithStopLoss:
    def test_fixed_stop_loss_triggers(self):
        # Price drops 10% after buying
        prices = [10.0] * 5 + [9.0, 8.5, 8.0, 7.5, 7.0]
        config = StopLossConfig(
            stop_loss_type=StopLossType.FIXED_PERCENT,
            stop_loss_pct=0.05,
            take_profit_pct=None,
        )
        engine = BacktestEngine(
            initial_capital=100000, commission_rate=0.0, slippage_rate=0.0,
            stop_loss_config=config,
        )
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        portfolio = engine.run()
        # Should have sold due to stop loss
        sell_trades = [t for t in portfolio.trade_history if t.side == OrderSide.SELL]
        assert len(sell_trades) >= 1

    def test_take_profit_triggers(self):
        # Price rises 15% after buying
        prices = [10.0] + [10.0 + i * 0.5 for i in range(1, 15)]
        config = StopLossConfig(
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
        )
        engine = BacktestEngine(
            initial_capital=100000, commission_rate=0.0, slippage_rate=0.0,
            stop_loss_config=config,
        )
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        portfolio = engine.run()
        sell_trades = [t for t in portfolio.trade_history if t.side == OrderSide.SELL]
        assert len(sell_trades) >= 1

class TestBacktestWithRiskManager:
    def test_risk_manager_blocks_order(self):
        prices = [10.0] * 10
        risk_manager = RiskManager([
            RiskRule(RiskRuleType.MAX_SINGLE_ORDER_VALUE, threshold=0.01),
        ])
        engine = BacktestEngine(
            initial_capital=100000, commission_rate=0.0, slippage_rate=0.0,
            risk_manager=risk_manager,
        )
        engine.set_strategy(AlwaysBuyStrategy())
        engine.add_data("600000", _make_test_data(prices))
        portfolio = engine.run()
        # All buy orders should be blocked by risk manager
        assert len(portfolio.trade_history) == 0
        assert portfolio.cash == 100000

class TestBaseStrategy:
    def test_calculate_max_buyable(self):
        strategy = AlwaysBuyStrategy()
        assert strategy.calculate_max_buyable(10.0, 1000.0) == 100
        assert strategy.calculate_max_buyable(10.0, 999.0) == 0
        assert strategy.calculate_max_buyable(10.0, 2500.0) == 200
        assert strategy.calculate_max_buyable(0.0, 1000.0) == 0
        assert strategy.calculate_max_buyable(-1.0, 1000.0) == 0

    def test_bar_history(self):
        strategy = AlwaysBuyStrategy()
        strategy.initialize(Portfolio(initial_capital=100000, cash=100000))
        bar = Bar(
            symbol="600000", datetime=datetime(2025, 1, 1),
            open=10.0, high=11.0, low=9.5, close=10.5, volume=1000000,
        )
        strategy._record_bar(bar)
        assert strategy.get_bar_count("600000") == 1
        assert strategy.get_close_series("600000").iloc[0] == 10.5
        assert strategy.get_high_series("600000").iloc[0] == 11.0
        assert strategy.get_low_series("600000").iloc[0] == 9.5
        assert strategy.get_volume_series("600000").iloc[0] == 1000000

    def test_create_orders(self):
        strategy = AlwaysBuyStrategy()
        bar = Bar(
            symbol="600000", datetime=datetime(2025, 1, 1),
            open=10.0, high=11.0, low=9.5, close=10.5, volume=1000000,
        )
        strategy._record_bar(bar)
        buy_order = strategy.create_buy_order("600000", 10.0, 100)
        assert buy_order.side == OrderSide.BUY
        assert buy_order.price == 10.0
        assert buy_order.quantity == 100

        sell_order = strategy.create_sell_order("600000", 12.0, 100)
        assert sell_order.side == OrderSide.SELL
