"""核心数据模型单元测试"""
import pytest
from datetime import datetime

from openquant.core.models import (
    MarketType, AssetType, FrequencyType, OrderSide, OrderStatus,
    Bar, Order, Position, TradeRecord, Portfolio,
)

class TestBar:
    def test_create_bar(self):
        bar = Bar(
            symbol="600000", datetime=datetime(2025, 1, 1),
            open=10.0, high=11.0, low=9.5, close=10.5,
            volume=1000000,
        )
        assert bar.symbol == "600000"
        assert bar.close == 10.5
        assert bar.market == MarketType.A_SHARE
        assert bar.frequency == FrequencyType.DAILY

    def test_bar_to_dict(self):
        bar = Bar(
            symbol="00700", datetime=datetime(2025, 6, 1),
            open=300.0, high=310.0, low=295.0, close=305.0,
            volume=5000000, amount=1500000000.0,
            market=MarketType.HK_STOCK,
        )
        result = bar.to_dict()
        assert result["symbol"] == "00700"
        assert result["market"] == "hk_stock"
        assert result["frequency"] == "daily"
        assert isinstance(result["datetime"], str)

class TestOrder:
    def test_create_buy_order(self):
        order = Order(
            order_id="test001", symbol="600000",
            side=OrderSide.BUY, price=10.0, quantity=100,
            created_at=datetime(2025, 1, 1),
        )
        assert order.status == OrderStatus.PENDING
        assert order.filled_price == 0.0
        assert order.filled_quantity == 0
        assert order.commission == 0.0

    def test_create_sell_order(self):
        order = Order(
            order_id="test002", symbol="600000",
            side=OrderSide.SELL, price=11.0, quantity=100,
            created_at=datetime(2025, 1, 2),
            market=MarketType.HK_STOCK,
        )
        assert order.side == OrderSide.SELL
        assert order.market == MarketType.HK_STOCK

class TestPosition:
    def test_market_value(self):
        pos = Position(symbol="600000", quantity=1000, avg_cost=10.0, current_price=12.0)
        assert pos.market_value == 12000.0

    def test_unrealized_pnl(self):
        pos = Position(symbol="600000", quantity=1000, avg_cost=10.0, current_price=12.0)
        assert pos.unrealized_pnl == 2000.0

    def test_unrealized_pnl_pct(self):
        pos = Position(symbol="600000", quantity=1000, avg_cost=10.0, current_price=12.0)
        assert abs(pos.unrealized_pnl_pct - 0.2) < 1e-9

    def test_unrealized_pnl_pct_zero_cost(self):
        pos = Position(symbol="600000", quantity=1000, avg_cost=0.0, current_price=12.0)
        assert pos.unrealized_pnl_pct == 0.0

    def test_negative_pnl(self):
        pos = Position(symbol="600000", quantity=500, avg_cost=20.0, current_price=15.0)
        assert pos.unrealized_pnl == -2500.0
        assert abs(pos.unrealized_pnl_pct - (-0.25)) < 1e-9

class TestTradeRecord:
    def test_buy_net_amount(self):
        trade = TradeRecord(
            trade_id="t001", order_id="o001", symbol="600000",
            side=OrderSide.BUY, price=10.0, quantity=100,
            commission=3.0, traded_at=datetime(2025, 1, 1),
        )
        # BUY: sign=1, net_amount = 1 * 10 * 100 + 3 = 1003
        assert trade.net_amount == 1003.0

    def test_sell_net_amount(self):
        trade = TradeRecord(
            trade_id="t002", order_id="o002", symbol="600000",
            side=OrderSide.SELL, price=12.0, quantity=100,
            commission=3.6, traded_at=datetime(2025, 1, 2),
        )
        # SELL: sign=-1, net_amount = -1 * 12 * 100 + 3.6 = -1196.4
        assert abs(trade.net_amount - (-1196.4)) < 1e-9

class TestPortfolio:
    def test_empty_portfolio(self):
        portfolio = Portfolio(initial_capital=100000.0, cash=100000.0)
        assert portfolio.total_market_value == 0.0
        assert portfolio.total_equity == 100000.0
        assert portfolio.total_return == 0.0
        assert portfolio.total_commission == 0.0

    def test_portfolio_with_positions(self):
        portfolio = Portfolio(initial_capital=100000.0, cash=50000.0)
        portfolio.positions["600000"] = Position(
            symbol="600000", quantity=1000, avg_cost=10.0, current_price=12.0,
        )
        portfolio.positions["00700"] = Position(
            symbol="00700", quantity=200, avg_cost=300.0, current_price=350.0,
        )
        assert portfolio.total_market_value == 12000.0 + 70000.0
        assert portfolio.total_equity == 50000.0 + 82000.0

    def test_portfolio_total_return(self):
        portfolio = Portfolio(initial_capital=100000.0, cash=120000.0)
        assert abs(portfolio.total_return - 0.2) < 1e-9

    def test_portfolio_zero_capital(self):
        portfolio = Portfolio(initial_capital=0.0, cash=0.0)
        assert portfolio.total_return == 0.0

    def test_portfolio_total_commission(self):
        portfolio = Portfolio(initial_capital=100000.0, cash=100000.0)
        portfolio.trade_history.append(
            TradeRecord(
                trade_id="t1", order_id="o1", symbol="600000",
                side=OrderSide.BUY, price=10.0, quantity=100,
                commission=3.0, traded_at=datetime(2025, 1, 1),
            )
        )
        portfolio.trade_history.append(
            TradeRecord(
                trade_id="t2", order_id="o2", symbol="600000",
                side=OrderSide.SELL, price=12.0, quantity=100,
                commission=3.6, traded_at=datetime(2025, 1, 2),
            )
        )
        assert abs(portfolio.total_commission - 6.6) < 1e-9

class TestEnums:
    def test_market_type_values(self):
        assert MarketType.A_SHARE.value == "a_share"
        assert MarketType.HK_STOCK.value == "hk_stock"
        assert MarketType.US_STOCK.value == "us_stock"

    def test_order_side_values(self):
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_status_values(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.REJECTED.value == "rejected"
