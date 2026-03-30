"""核心数据模型定义

包含市场类型、资产类型、K线数据、订单、持仓、组合等核心数据结构。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class MarketType(enum.Enum):
    """市场类型"""
    A_SHARE = "a_share"
    HK_STOCK = "hk_stock"
    US_STOCK = "us_stock"
    FUND = "fund"
    FUTURES = "futures"
    CRYPTO = "crypto"


class AssetType(enum.Enum):
    """资产类型"""
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    FUND = "fund"
    BOND = "bond"
    FUTURES = "futures"
    OPTION = "option"


class FrequencyType(enum.Enum):
    """K线频率"""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    MINUTE_60 = "60min"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class EventType(enum.Enum):
    """事件类型"""
    EARNINGS = "earnings"              # 财报发布
    DIVIDEND = "dividend"              # 分红派息
    BLOCK_TRADE = "block_trade"        # 大宗交易
    SHAREHOLDER = "shareholder"        # 股东增减持
    INSIDER = "insider"                # 内部人交易
    ANALYST_UPGRADE = "analyst_upgrade"    # 分析师上调评级
    ANALYST_DOWNGRADE = "analyst_downgrade"  # 分析师下调评级
    NEWS_POSITIVE = "news_positive"    # 正面新闻/舆情
    NEWS_NEGATIVE = "news_negative"    # 负面新闻/舆情
    POLICY = "policy"                  # 政策/监管事件
    BUYBACK = "buyback"                # 股票回购
    CUSTOM = "custom"                  # 自定义事件


class EventSentiment(enum.Enum):
    """事件情绪方向"""
    BULLISH = "bullish"      # 利多
    BEARISH = "bearish"      # 利空
    NEUTRAL = "neutral"      # 中性


class OrderSide(enum.Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(enum.Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Bar:
    """K线数据"""
    symbol: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float = 0.0
    turnover_rate: float = 0.0
    market: MarketType = MarketType.A_SHARE
    frequency: FrequencyType = FrequencyType.DAILY

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "datetime": self.datetime.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "amount": self.amount,
            "turnover_rate": self.turnover_rate,
            "market": self.market.value,
            "frequency": self.frequency.value,
        }


@dataclass
class Order:
    """交易订单"""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: int
    created_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    filled_at: datetime | None = None
    commission: float = 0.0
    slippage: float = 0.0
    market: MarketType = MarketType.A_SHARE


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    market: MarketType = MarketType.A_SHARE

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class TradeRecord:
    """成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: int
    commission: float
    traded_at: datetime
    market: MarketType = MarketType.A_SHARE

    @property
    def net_amount(self) -> float:
        sign = 1 if self.side == OrderSide.BUY else -1
        return sign * self.price * self.quantity + self.commission


@dataclass
class Portfolio:
    """投资组合"""
    initial_capital: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trade_history: list[TradeRecord] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_market_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.cash + self.total_market_value

    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_equity - self.initial_capital) / self.initial_capital

    @property
    def total_commission(self) -> float:
        return sum(trade.commission for trade in self.trade_history)


@dataclass
class EventFactor:
    """事件因子

    表示一个影响股价的离散事件，如财报发布、分红、大宗交易、新闻舆情等。
    策略可根据事件的类型、情绪方向和强度来调整交易决策。
    """
    symbol: str
    event_date: datetime
    event_type: EventType
    sentiment: EventSentiment
    strength: float = 1.0
    description: str = ""
    source: str = ""
    market: MarketType = MarketType.A_SHARE
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "event_date": self.event_date.isoformat(),
            "event_type": self.event_type.value,
            "sentiment": self.sentiment.value,
            "strength": self.strength,
            "description": self.description,
            "source": self.source,
            "market": self.market.value,
        }
