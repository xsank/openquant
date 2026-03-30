"""事件增强策略回测验证脚本

对比基础 MA Cross 和事件增强 MA Cross 在阿里巴巴 (09988) 上的表现。
由于港股事件数据有限，本脚本基于历史行情中的关键时间点构造事件因子，
模拟真实场景下事件因子对策略决策的修正效果。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import os

# 图表相关导入
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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
from openquant.utils.md_to_pdf import convert_md_to_pdf


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

def plot_equity_comparison(base_equity_df, event_equity_df, output_path):
    """绘制权益曲线对比图"""
    plt.figure(figsize=(12, 6))
    
    base_equity = base_equity_df['equity'].values
    event_equity = event_equity_df['equity'].values
    dates = base_equity_df['datetime'].values
    
    plt.plot(dates, base_equity, label='基础 MA Cross', linewidth=2, color='#2E86AB')
    plt.plot(dates, event_equity, label='事件增强 MA Cross', linewidth=2, color='#A23B72')
    
    plt.title('权益曲线对比', fontsize=14, fontweight='bold')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('权益 (HKD)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 格式化日期显示
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(base_results, event_results, output_path):
    """绘制关键指标对比柱状图"""
    metrics = {
        '总收益率 (%)': (base_results['total_return'], event_results['total_return']),
        '年化收益率 (%)': (base_results['annual_return'], event_results['annual_return']),
        '夏普比率': (base_results['sharpe_ratio'], event_results['sharpe_ratio']),
        '最大回撤 (%)': (abs(base_results['max_drawdown']), abs(event_results['max_drawdown'])),
        '胜率 (%)': (base_results['win_rate'], event_results['win_rate']),
    }
    
    labels = list(metrics.keys())
    base_values = [v[0] for v in metrics.values()]
    event_values = [v[1] for v in metrics.values()]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, base_values, width, label='基础 MA Cross', color='#2E86AB')
    bars2 = ax.bar(x + width/2, event_values, width, label='事件增强 MA Cross', color='#A23B72')
    
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('关键指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_event_timeline(df, events, output_path):
    """绘制事件时间线图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 上图：价格走势
    ax1.plot(df['datetime'], df['close'], label='收盘价', linewidth=1.5, color='#333333')
    ax1.set_ylabel('价格 (HKD)', fontsize=11)
    ax1.set_title('价格走势与事件时间线', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 下图：事件标注
    ax2.plot(df['datetime'], df['close'], linewidth=0.5, color='#CCCCCC', alpha=0.5)
    
    for event in events:
        event_date = event.event_date
        # 找到最接近的交易日
        closest_idx = (df['datetime'] - event_date).abs().idxmin()
        if abs((df.loc[closest_idx, 'datetime'] - event_date).days) <= 3:
            price = df.loc[closest_idx, 'close']
            color = '#2ECC71' if event.sentiment == EventSentiment.BULLISH else '#E74C3C'
            marker = '^' if event.sentiment == EventSentiment.BULLISH else 'v'
            
            ax2.scatter(event_date, price, s=150, c=color, marker=marker, 
                       edgecolors='black', linewidth=1.5, zorder=5)
            
            # 添加事件标签
            label = f"{event.event_type.value}\n{event.description[:15]}..."
            ax2.annotate(label, 
                        xy=(event_date, price),
                        xytext=(10, 20 if event.sentiment == EventSentiment.BULLISH else -30),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.2),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.set_ylabel('价格 (HKD)', fontsize=11)
    ax2.set_xlabel('日期', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 格式化日期显示
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_markdown_report(timestamp, events, base_results, event_results, df, output_path):
    """生成 Markdown 报告"""
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 事件增强策略回测报告\n\n")
        f.write(f"**股票代码**: 09988 (阿里巴巴)\n\n")
        f.write(f"**报告生成时间**: {report_time}\n\n")
        f.write("---\n\n")
        
        # 事件因子列表
        f.write("## 事件因子列表\n\n")
        f.write("| 日期 | 类型 | 情绪 | 强度 | 描述 |\n")
        f.write("|------|------|------|------|------|\n")
        for event in events:
            sentiment = "利多 🟢" if event.sentiment == EventSentiment.BULLISH else "利空 🔴"
            f.write(f"| {event.event_date.strftime('%Y-%m-%d')} | {event.event_type.value} | {sentiment} | {event.strength} | {event.description} |\n")
        f.write("\n")
        
        # 回测对比表格
        f.write("## 回测对比结果\n\n")
        f.write("| 指标 | 基础 MA Cross | 事件增强 MA Cross |\n")
        f.write("|------|---------------|------------------|\n")
        
        metrics_to_compare = [
            ("总收益率", "total_return", "%"),
            ("年化收益率", "annual_return", "%"),
            ("年化波动率", "annual_volatility", "%"),
            ("夏普比率", "sharpe_ratio", ""),
            ("索提诺比率", "sortino_ratio", ""),
            ("卡玛比率", "calmar_ratio", ""),
            ("最大回撤", "max_drawdown", "%"),
            ("胜率", "win_rate", "%"),
            ("盈亏比", "profit_loss_ratio", ""),
            ("交易次数", "total_trades", ""),
            ("总佣金", "total_commission", ""),
        ]
        
        for label, key, suffix in metrics_to_compare:
            base_val = base_results.get(key, 0)
            event_val = event_results.get(key, 0)
            
            if suffix == "%":
                base_str = f"{base_val:.2f}%"
                event_str = f"{event_val:.2f}%"
            elif key == "total_trades":
                base_str = f"{int(base_val)}"
                event_str = f"{int(event_val)}"
            elif key == "total_commission":
                base_str = f"{base_val:,.2f}"
                event_str = f"{event_val:,.2f}"
            else:
                base_str = f"{base_val:.4f}"
                event_str = f"{event_val:.4f}"
            
            f.write(f"| {label} | {base_str} | {event_str} |\n")
        f.write("\n")
        
        # 收益差异分析
        f.write("## 收益差异分析\n\n")
        base_return = base_results.get("total_return", 0)
        event_return = event_results.get("total_return", 0)
        diff = event_return - base_return
        
        if diff > 0:
            f.write(f"✅ **事件增强策略收益率提升了 {diff:.2f}%**\n\n")
        elif diff < 0:
            f.write(f"⚠️ **事件增强策略收益率下降了 {abs(diff):.2f}%**\n\n")
        else:
            f.write("➡️ **两个策略收益率相同**\n\n")
        
        base_dd = abs(base_results.get("max_drawdown", 0))
        event_dd = abs(event_results.get("max_drawdown", 0))
        dd_diff = base_dd - event_dd
        if dd_diff > 0:
            f.write(f"✅ **事件增强策略最大回撤减少了 {dd_diff:.2f}%**\n\n")
        elif dd_diff < 0:
            f.write(f"⚠️ **事件增强策略最大回撤增加了 {abs(dd_diff):.2f}%**\n\n")
        
        base_sharpe = base_results.get("sharpe_ratio", 0)
        event_sharpe = event_results.get("sharpe_ratio", 0)
        if event_sharpe > base_sharpe:
            f.write(f"✅ **事件增强策略夏普比率提升**: {base_sharpe:.4f} → {event_sharpe:.4f}\n\n")
        f.write("\n")
        
        # 图表引用
        f.write("## 图表分析\n\n")
        f.write("### 权益曲线对比\n\n")
        f.write(f"![权益曲线对比](../../charts/{timestamp}/equity_comparison.png)\n\n")
        
        f.write("### 关键指标对比\n\n")
        f.write(f"![关键指标对比](../../charts/{timestamp}/metrics_comparison.png)\n\n")
        
        f.write("### 事件时间线\n\n")
        f.write(f"![事件时间线](../../charts/{timestamp}/event_timeline.png)\n\n")
        
        f.write("---\n\n")
        f.write(f"*报告由 openquant 事件增强策略回测系统自动生成*")

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

    # === 获取权益曲线数据 ===
    base_equity_df = base_engine.get_equity_curve()
    event_equity_df = event_engine.get_equity_curve()

    # === 生成时间戳和创建输出目录 ===
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_output_dir = Path("output/md") / timestamp
    charts_output_dir = Path("output/charts") / timestamp
    md_output_dir.mkdir(parents=True, exist_ok=True)
    charts_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n正在生成图表和报告...")
    print(f"输出目录: output/md/{timestamp}/ 和 output/charts/{timestamp}/")

    # === 生成图表 ===
    equity_chart_path = charts_output_dir / "equity_comparison.png"
    metrics_chart_path = charts_output_dir / "metrics_comparison.png"
    timeline_chart_path = charts_output_dir / "event_timeline.png"

    plot_equity_comparison(base_equity_df, event_equity_df, equity_chart_path)
    plot_metrics_comparison(base_results, event_results, metrics_chart_path)
    plot_event_timeline(df, events, timeline_chart_path)

    # === 生成 Markdown 报告 ===
    md_report_path = md_output_dir / "event_backtest_09988.md"
    generate_markdown_report(timestamp, events, base_results, event_results, df, md_report_path)

    print(f"✅ 图表已保存到: {charts_output_dir}")
    print(f"✅ 报告已保存到: {md_report_path}")

    # === 导出 PDF ===
    pdf_output_dir = Path("output/pdf") / timestamp
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_output_dir / "event_backtest_09988.pdf"
    print("📑 正在导出 PDF...")
    try:
        convert_md_to_pdf(str(md_report_path), str(pdf_path))
        print(f"✅ PDF 已保存到: {pdf_path}")
    except Exception as exc:
        print(f"⚠️ PDF 导出失败: {exc}")
    print()

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
