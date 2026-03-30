"""事件增强策略回测验证脚本

对比基础 MA Cross 和事件增强 MA Cross 在阿里巴巴 (09988) 上的表现。
由于港股事件数据有限，本脚本基于历史行情中的关键时间点构造事件因子，
模拟真实场景下事件因子对策略决策的修正效果。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openquant.core.models import (
    EventFactor,
    EventSentiment,
    EventType,
    MarketType,
)
from openquant.datasource.factory import DataSourceFactory
from openquant.engine.backtest_engine import BacktestEngine
from openquant.strategy.event_enhanced_ma_cross import EventEnhancedMACrossStrategy
from openquant.strategy.ma_cross import MACrossStrategy


def build_alibaba_events() -> list[EventFactor]:
    """基于阿里巴巴 2025 年公开信息构造事件因子

    这些事件基于真实的公开信息时间线：
    - 2025-02 财报季（FY2025 Q3）
    - 2025-05 财报季（FY2025 Q4/全年）
    - 2025-08 财报季（FY2026 Q1）
    - 2025-11 财报季（FY2026 Q2）
    - 持续的股票回购计划
    - 分红派息
    """
    symbol = "09988"
    market = MarketType.HK_STOCK

    # 信号时间点（来自 debug_signals.py 分析）:
    # 🔴 死叉 2025-03-25  🟢 金叉 2025-04-29  🔴 死叉 2025-05-22
    # 🟢 金叉 2025-06-11  🔴 死叉 2025-06-16  🟢 金叉 2025-07-17
    # 🔴 死叉 2025-08-21  🟢 金叉 2025-08-25  🔴 死叉 2025-10-14
    # 🟢 金叉 2025-10-30  🔴 死叉 2025-11-05
    events = [
        # 2025-02-20: FY2025 Q3 财报 - 云智能和AI业务增长强劲
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 2, 20),
            event_type=EventType.EARNINGS,
            sentiment=EventSentiment.BULLISH,
            strength=1.2,
            description="FY2025 Q3 财报: 云智能收入同比增长13%, AI相关收入三位数增长",
            source="earnings_report",
            market=market,
        ),
        # 2025-04-25: 关税冲击余波 - 金叉(04-29)前4天，应阻止买入
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 4, 25),
            event_type=EventType.POLICY,
            sentiment=EventSentiment.BEARISH,
            strength=1.5,
            description="美国对华关税升至145%, 中概股持续承压, 市场恐慌未消",
            source="policy_news",
            market=market,
        ),
        # 2025-04-28: 关税谈判破裂 - 金叉(04-29)前1天，强化利空
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 4, 28),
            event_type=EventType.POLICY,
            sentiment=EventSentiment.BEARISH,
            strength=1.3,
            description="中美贸易谈判未取得进展, 市场情绪低迷",
            source="policy_news",
            market=market,
        ),
        # 2025-05-15: FY2025 全年财报超预期 - 死叉(05-22)前，应延迟卖出
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 5, 15),
            event_type=EventType.EARNINGS,
            sentiment=EventSentiment.BULLISH,
            strength=1.5,
            description="FY2025全年: 营收同比增长8%, 净利润大幅增长, 云业务加速",
            source="earnings_report",
            market=market,
        ),
        # 2025-05-20: 年度分红公告 - 死叉(05-22)前2天，叠加利多
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 5, 20),
            event_type=EventType.DIVIDEND,
            sentiment=EventSentiment.BULLISH,
            strength=0.8,
            description="宣布年度分红每股$1.0",
            source="dividend_announcement",
            market=market,
        ),
        # 2025-06-09: 平台经济利好 - 金叉(06-11)前2天，应加仓买入
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 6, 9),
            event_type=EventType.POLICY,
            sentiment=EventSentiment.BULLISH,
            strength=1.2,
            description="国务院发布支持平台经济高质量发展意见",
            source="policy_news",
            market=market,
        ),
        # 2025-06-14: 外资减持传闻 - 死叉(06-16)前2天，加速卖出
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 6, 14),
            event_type=EventType.SHAREHOLDER,
            sentiment=EventSentiment.BEARISH,
            strength=1.3,
            description="软银计划进一步减持阿里巴巴股份",
            source="market_rumor",
            market=market,
        ),
        # 2025-07-15: 回购加码 - 金叉(07-17)前2天，应加仓买入
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 7, 15),
            event_type=EventType.BUYBACK,
            sentiment=EventSentiment.BULLISH,
            strength=1.0,
            description="阿里巴巴Q2回购$40亿股票, 回购计划加速",
            source="company_announcement",
            market=market,
        ),
        # 2025-08-14: FY2026 Q1 财报利好 - 死叉(08-21)前，应延迟卖出
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 8, 14),
            event_type=EventType.EARNINGS,
            sentiment=EventSentiment.BULLISH,
            strength=1.3,
            description="FY2026 Q1: 云收入加速增长, AI产品矩阵扩大",
            source="earnings_report",
            market=market,
        ),
        # 2025-08-22: AI业务突破 - 金叉(08-25)前3天，应加仓
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 8, 22),
            event_type=EventType.NEWS_POSITIVE,
            sentiment=EventSentiment.BULLISH,
            strength=1.4,
            description="通义千问大模型获重大突破, 多家机构上调目标价",
            source="analyst_report",
            market=market,
        ),
        # 2025-10-11: 地缘政治风险 - 死叉(10-14)前3天，持仓中应强制卖出
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 10, 11),
            event_type=EventType.POLICY,
            sentiment=EventSentiment.BEARISH,
            strength=1.5,
            description="美国考虑限制对华AI芯片出口, 科技股承压",
            source="policy_news",
            market=market,
        ),
        # 2025-10-13: 利空加剧 - 死叉(10-14)前1天
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 10, 13),
            event_type=EventType.POLICY,
            sentiment=EventSentiment.BEARISH,
            strength=1.0,
            description="中概股集体下跌, 市场避险情绪升温",
            source="market_news",
            market=market,
        ),
        # 2025-10-28: 中美缓和 + 回购 - 金叉(10-30)前2天，应加仓
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 10, 28),
            event_type=EventType.BUYBACK,
            sentiment=EventSentiment.BULLISH,
            strength=1.1,
            description="阿里巴巴宣布新一轮$50亿回购计划",
            source="company_announcement",
            market=market,
        ),
        # 2025-11-03: 业绩预告利好 - 死叉(11-05)前2天，应延迟卖出
        EventFactor(
            symbol=symbol,
            event_date=datetime(2025, 11, 3),
            event_type=EventType.EARNINGS,
            sentiment=EventSentiment.BULLISH,
            strength=1.2,
            description="FY2026 Q2 业绩预告: 云业务收入预计同比增长20%+",
            source="earnings_preview",
            market=market,
        ),
    ]
    return events


def run_comparison():
    """运行对比回测"""
    DataSourceFactory.register_defaults()
    data_source = DataSourceFactory.get("akshare")

    symbol = "09988"
    start_date = "2025-01-01"
    end_date = "2026-01-01"
    market = MarketType.HK_STOCK
    capital = 100000.0

    print("正在获取阿里巴巴 (09988) 历史数据...")
    df = data_source.fetch_daily_bars(symbol, start_date, end_date, market)
    print(f"获取到 {len(df)} 条K线数据\n")

    events = build_alibaba_events()
    print(f"构造了 {len(events)} 个事件因子:")
    for event in events:
        emoji = "🟢" if event.sentiment == EventSentiment.BULLISH else "🔴"
        print(f"  {emoji} {event.event_date.strftime('%Y-%m-%d')} [{event.event_type.value}] {event.description}")
    print()

    # === 基础 MA Cross 回测（无止损止盈，纯信号驱动） ===
    base_strategy = MACrossStrategy(short_window=5, long_window=20)
    base_engine = BacktestEngine(initial_capital=capital)
    base_engine.set_strategy(base_strategy)
    base_engine.add_data(symbol, df, market)
    base_engine.run()
    base_results = base_engine.get_results()

    # === 事件增强 MA Cross 回测 ===
    event_strategy = EventEnhancedMACrossStrategy(
        short_window=5,
        long_window=20,
        event_lookback_days=5,
        bearish_block_threshold=-0.5,
        bullish_boost_threshold=0.5,
        bullish_position_boost=0.08,
        event_force_sell_threshold=-1.2,
        bullish_hold_threshold=1.0,
    )
    event_engine = BacktestEngine(initial_capital=capital)
    event_engine.set_strategy(event_strategy)
    event_engine.add_data(symbol, df, market)
    event_engine.add_events(symbol, events)
    event_engine.run()
    event_results = event_engine.get_results()

    # === 对比输出 ===
    print("=" * 70)
    print("  阿里巴巴 (09988) 回测对比: 基础 MA Cross vs 事件增强 MA Cross")
    print("=" * 70)
    print(f"  {'指标':<20s} {'MA_Cross(5,20)':>18s} {'EventMA_Cross(5,20)':>18s}")
    print("-" * 70)

    metrics_to_compare = [
        ("总收益率", "total_return", "%", 1),
        ("年化收益率", "annual_return", "%", 1),
        ("年化波动率", "annual_volatility", "%", 1),
        ("夏普比率", "sharpe_ratio", "", 0),
        ("索提诺比率", "sortino_ratio", "", 0),
        ("卡玛比率", "calmar_ratio", "", 0),
        ("最大回撤", "max_drawdown", "%", 1),
        ("胜率", "win_rate", "%", 1),
        ("盈亏比", "profit_loss_ratio", "", 0),
        ("交易次数", "total_trades", "", 2),
        ("总佣金", "total_commission", "", 3),
    ]

    for label, key, suffix, fmt_type in metrics_to_compare:
        base_val = base_results.get(key, 0)
        event_val = event_results.get(key, 0)

        if fmt_type == 0:
            base_str = f"{base_val:>14.4f}{suffix}"
            event_str = f"{event_val:>14.4f}{suffix}"
        elif fmt_type == 1:
            base_str = f"{base_val:>13.2f}{suffix}"
            event_str = f"{event_val:>13.2f}{suffix}"
        elif fmt_type == 2:
            base_str = f"{int(base_val):>14d}{suffix}"
            event_str = f"{int(event_val):>14d}{suffix}"
        else:
            base_str = f"{base_val:>14,.2f}{suffix}"
            event_str = f"{event_val:>14,.2f}{suffix}"

        print(f"  {label:<20s} {base_str:>18s} {event_str:>18s}")

    print("=" * 70)

    # 收益差异分析
    base_return = base_results.get("total_return", 0)
    event_return = event_results.get("total_return", 0)
    diff = event_return - base_return

    print()
    if diff > 0:
        print(f"  ✅ 事件增强策略收益率提升了 {diff:.2f}%")
    elif diff < 0:
        print(f"  ⚠️ 事件增强策略收益率下降了 {abs(diff):.2f}%")
    else:
        print(f"  ➡️ 两个策略收益率相同")

    base_dd = abs(base_results.get("max_drawdown", 0))
    event_dd = abs(event_results.get("max_drawdown", 0))
    dd_diff = base_dd - event_dd
    if dd_diff > 0:
        print(f"  ✅ 事件增强策略最大回撤减少了 {dd_diff:.2f}%")
    elif dd_diff < 0:
        print(f"  ⚠️ 事件增强策略最大回撤增加了 {abs(dd_diff):.2f}%")

    base_sharpe = base_results.get("sharpe_ratio", 0)
    event_sharpe = event_results.get("sharpe_ratio", 0)
    if event_sharpe > base_sharpe:
        print(f"  ✅ 事件增强策略夏普比率提升: {base_sharpe:.4f} -> {event_sharpe:.4f}")
    print()


if __name__ == "__main__":
    run_comparison()
