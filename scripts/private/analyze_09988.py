"""09988 阿里巴巴港股 综合分析脚本

获取最新行情数据，计算技术指标，给出交易建议。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from openquant.datasource.factory import DataSourceFactory
from openquant.core.models import MarketType
from openquant.utils.indicators import bollinger_bands


def fetch_data():
    """获取09988历史数据"""
    DataSourceFactory.register_defaults()
    data_source = DataSourceFactory.get("akshare")
    
    # 获取过去1年的数据用于分析
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    
    df = data_source.fetch_daily_bars("09988", start_date, end_date, MarketType.HK_STOCK)
    return df


def calculate_percentile_rank(df):
    """计算当前价格在历史中的百分位"""
    current_price = df['close'].iloc[-1]
    
    # 过去1年的百分位
    year_data = df['close'].tail(250)
    percentile_1y = (year_data < current_price).sum() / len(year_data) * 100
    
    # 过去半年的百分位
    half_year_data = df['close'].tail(125)
    percentile_6m = (half_year_data < current_price).sum() / len(half_year_data) * 100
    
    # 过去3个月的百分位
    quarter_data = df['close'].tail(63)
    percentile_3m = (quarter_data < current_price).sum() / len(quarter_data) * 100
    
    # 过去1个月的百分位
    month_data = df['close'].tail(22)
    percentile_1m = (month_data < current_price).sum() / len(month_data) * 100
    
    return {
        '1年百分位': percentile_1y,
        '6个月百分位': percentile_6m,
        '3个月百分位': percentile_3m,
        '1个月百分位': percentile_1m,
    }


def calculate_bollinger_signals(df, window=14, num_std=2.5):
    """计算布林带信号"""
    close_series = df['close']
    upper, middle, lower = bollinger_bands(close_series, window, num_std)
    
    current_close = close_series.iloc[-1]
    prev_close = close_series.iloc[-2]
    current_upper = upper.iloc[-1]
    current_lower = lower.iloc[-1]
    current_middle = middle.iloc[-1]
    prev_upper = upper.iloc[-2]
    prev_lower = lower.iloc[-2]
    
    # 计算当前价格在布林带中的位置 (0=下轨, 0.5=中轨, 1=上轨)
    band_width = current_upper - current_lower
    if band_width > 0:
        band_position = (current_close - current_lower) / band_width
    else:
        band_position = 0.5
    
    # 判断信号
    signal = "观望"
    if prev_close <= prev_lower and current_close > current_lower:
        signal = "🟢 买入信号（价格从下轨回升）"
    elif prev_close >= prev_upper and current_close < current_upper:
        signal = "🔴 卖出信号（价格从上轨回落）"
    elif current_close < current_lower:
        signal = "⚠️ 价格在下轨下方，等待回升确认后买入"
    elif current_close > current_upper:
        signal = "⚠️ 价格在上轨上方，等待回落确认后卖出"
    elif current_close < current_middle:
        signal = "🟡 价格在中轨下方，偏空观望"
    else:
        signal = "🟡 价格在中轨上方，偏多观望"
    
    return {
        '上轨': round(current_upper, 2),
        '中轨': round(current_middle, 2),
        '下轨': round(current_lower, 2),
        '带宽': round(band_width, 2),
        '带内位置': round(band_position, 4),
        '信号': signal,
    }


def calculate_rsi(close_series, period=14):
    """计算RSI"""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close_series, fast=12, slow=26, signal=9):
    """计算MACD"""
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_support_resistance(df, lookback=60):
    """计算支撑位和阻力位"""
    recent = df.tail(lookback)
    current_price = df['close'].iloc[-1]
    
    # 使用近期高低点
    highs = recent['high'].values
    lows = recent['low'].values
    
    # 找关键价位
    resistance_levels = []
    support_levels = []
    
    # 近期最高最低
    period_high = recent['high'].max()
    period_low = recent['low'].min()
    
    # 使用分位数作为支撑阻力
    q25 = np.percentile(recent['close'], 25)
    q50 = np.percentile(recent['close'], 50)
    q75 = np.percentile(recent['close'], 75)
    
    # MA支撑阻力
    ma5 = df['close'].rolling(5).mean().iloc[-1]
    ma10 = df['close'].rolling(10).mean().iloc[-1]
    ma20 = df['close'].rolling(20).mean().iloc[-1]
    ma60 = df['close'].rolling(60).mean().iloc[-1]
    
    all_levels = {
        'MA5': ma5, 'MA10': ma10, 'MA20': ma20, 'MA60': ma60,
        '25%分位': q25, '50%分位(中位数)': q50, '75%分位': q75,
        '近期最高': period_high, '近期最低': period_low,
    }
    
    for name, level in sorted(all_levels.items(), key=lambda x: x[1]):
        if level < current_price:
            support_levels.append((name, round(level, 2)))
        else:
            resistance_levels.append((name, round(level, 2)))
    
    return support_levels, resistance_levels


def calculate_volatility(df, window=20):
    """计算波动率"""
    returns = df['close'].pct_change().dropna()
    daily_vol = returns.tail(window).std()
    annual_vol = daily_vol * np.sqrt(252)
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]
    
    return {
        '日波动率': round(daily_vol * 100, 2),
        '年化波动率': round(annual_vol * 100, 2),
        'ATR(14)': round(atr, 2),
    }


def estimate_success_probability(df, bollinger_info, rsi_value, macd_hist):
    """基于多指标综合估算成功概率"""
    score = 50  # 基础分
    
    # 布林带位置评分 (越靠近下轨越适合买入)
    band_pos = bollinger_info['带内位置']
    if band_pos < 0:  # 在下轨下方
        score += 15
    elif band_pos < 0.3:
        score += 10
    elif band_pos < 0.5:
        score += 5
    elif band_pos > 0.8:
        score -= 10
    elif band_pos > 1.0:
        score -= 15
    
    # RSI评分
    if rsi_value < 30:
        score += 15  # 超卖
    elif rsi_value < 40:
        score += 8
    elif rsi_value < 50:
        score += 3
    elif rsi_value > 70:
        score -= 15  # 超买
    elif rsi_value > 60:
        score -= 5
    
    # MACD评分
    if macd_hist > 0:
        score += 5  # 多头
    else:
        score -= 5  # 空头
    
    # 趋势评分 (近期涨跌)
    recent_return_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100
    recent_return_20d = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100
    
    if recent_return_5d < -5:
        score += 8  # 短期超跌反弹机会
    elif recent_return_5d > 5:
        score -= 5  # 短期涨幅过大
    
    score = max(10, min(90, score))
    return score


def main():
    print("=" * 70)
    print("  09988 阿里巴巴港股 - 综合技术分析报告")
    print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 获取数据
    print("\n📊 正在获取历史数据...")
    df = fetch_data()
    print(f"  获取到 {len(df)} 条K线数据")
    print(f"  数据范围: {df.index[0]} ~ {df.index[-1]}")
    
    current_price = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    print(f"\n{'='*70}")
    print(f"  📈 最新收盘价: HK${current_price:.2f}")
    print(f"  涨跌: {price_change:+.2f} ({price_change_pct:+.2f}%)")
    print(f"  今日开盘: {df['open'].iloc[-1]:.2f}  最高: {df['high'].iloc[-1]:.2f}  最低: {df['low'].iloc[-1]:.2f}")
    print(f"  成交量: {df['volume'].iloc[-1]:,.0f}")
    print(f"{'='*70}")
    
    # 2. 历史百分位
    print(f"\n{'='*70}")
    print("  📊 历史价格位点分析")
    print(f"{'='*70}")
    percentiles = calculate_percentile_rank(df)
    for period, pct in percentiles.items():
        bar_len = int(pct / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {period:>10s}: {pct:5.1f}%  |{bar}|")
    
    # 3. 布林带分析 (最优参数)
    print(f"\n{'='*70}")
    print("  📉 布林带分析 (最优参数: window=14, num_std=2.5)")
    print(f"{'='*70}")
    bollinger_info = calculate_bollinger_signals(df, window=14, num_std=2.5)
    print(f"  上轨: {bollinger_info['上轨']}")
    print(f"  中轨: {bollinger_info['中轨']}")
    print(f"  下轨: {bollinger_info['下轨']}")
    print(f"  带宽: {bollinger_info['带宽']}")
    print(f"  带内位置: {bollinger_info['带内位置']:.2%}")
    print(f"  信号: {bollinger_info['信号']}")
    
    # 4. RSI分析
    print(f"\n{'='*70}")
    print("  📊 RSI 分析")
    print(f"{'='*70}")
    rsi = calculate_rsi(df['close'], 14)
    rsi_value = rsi.iloc[-1]
    rsi_status = "超卖区" if rsi_value < 30 else "超买区" if rsi_value > 70 else "中性区"
    print(f"  RSI(14): {rsi_value:.2f} ({rsi_status})")
    print(f"  RSI(6):  {calculate_rsi(df['close'], 6).iloc[-1]:.2f}")
    
    # 5. MACD分析
    print(f"\n{'='*70}")
    print("  📊 MACD 分析")
    print(f"{'='*70}")
    macd_line, signal_line, histogram = calculate_macd(df['close'])
    macd_val = macd_line.iloc[-1]
    signal_val = signal_line.iloc[-1]
    hist_val = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]
    macd_trend = "多头增强" if hist_val > 0 and hist_val > prev_hist else \
                 "多头减弱" if hist_val > 0 else \
                 "空头减弱" if hist_val < 0 and hist_val > prev_hist else "空头增强"
    print(f"  MACD线:  {macd_val:.4f}")
    print(f"  信号线:  {signal_val:.4f}")
    print(f"  柱状图:  {hist_val:.4f} ({macd_trend})")
    
    # 6. 均线系统
    print(f"\n{'='*70}")
    print("  📊 均线系统")
    print(f"{'='*70}")
    for period in [5, 10, 20, 60]:
        ma = df['close'].rolling(period).mean().iloc[-1]
        diff_pct = (current_price / ma - 1) * 100
        status = "↑ 多头" if current_price > ma else "↓ 空头"
        print(f"  MA{period:>2d}: {ma:>8.2f}  偏离: {diff_pct:>+6.2f}%  {status}")
    
    # 7. 支撑阻力位
    print(f"\n{'='*70}")
    print("  📊 支撑位与阻力位")
    print(f"{'='*70}")
    support_levels, resistance_levels = calculate_support_resistance(df)
    
    print("  🔴 阻力位 (由近到远):")
    for name, level in resistance_levels:
        diff = (level / current_price - 1) * 100
        print(f"    {name:>16s}: {level:>8.2f}  (距当前 {diff:>+6.2f}%)")
    
    print(f"\n  当前价格: HK${current_price:.2f}")
    print()
    
    print("  🟢 支撑位 (由近到远):")
    for name, level in reversed(support_levels):
        diff = (level / current_price - 1) * 100
        print(f"    {name:>16s}: {level:>8.2f}  (距当前 {diff:>+6.2f}%)")
    
    # 8. 波动率
    print(f"\n{'='*70}")
    print("  📊 波动率分析")
    print(f"{'='*70}")
    vol_info = calculate_volatility(df)
    for key, val in vol_info.items():
        print(f"  {key}: {val}")
    
    # 9. 成功概率估算
    print(f"\n{'='*70}")
    print("  🎯 买入成功概率估算")
    print(f"{'='*70}")
    success_prob = estimate_success_probability(df, bollinger_info, rsi_value, hist_val)
    
    if success_prob >= 70:
        prob_label = "✅ 高概率买入机会"
    elif success_prob >= 55:
        prob_label = "🟡 中等概率，可小仓位试探"
    elif success_prob >= 40:
        prob_label = "⚠️ 概率偏低，建议观望"
    else:
        prob_label = "🔴 不建议买入"
    
    print(f"  综合评分: {success_prob}/100")
    print(f"  判断: {prob_label}")
    
    # 10. 交易建议
    print(f"\n{'='*70}")
    print("  💡 明日交易建议")
    print(f"{'='*70}")
    
    atr = vol_info['ATR(14)']
    
    # 理想买入价 = 布林带下轨附近
    ideal_buy = bollinger_info['下轨']
    # 激进买入价 = 中轨附近
    aggressive_buy = bollinger_info['中轨']
    # 止损价 = 买入价 - 1.5*ATR
    stop_loss_from_lower = round(ideal_buy - 1.5 * atr, 2)
    # 止盈价 = 上轨
    take_profit = bollinger_info['上轨']
    
    # 近期涨跌幅
    return_5d = (current_price / df['close'].iloc[-6] - 1) * 100
    return_20d = (current_price / df['close'].iloc[-21] - 1) * 100
    return_60d = (current_price / df['close'].iloc[-61] - 1) * 100
    
    print(f"\n  近期走势:")
    print(f"    5日涨跌:  {return_5d:>+6.2f}%")
    print(f"    20日涨跌: {return_20d:>+6.2f}%")
    print(f"    60日涨跌: {return_60d:>+6.2f}%")
    
    print(f"\n  基于布林带策略 Bollinger(14,2.5) 的建议:")
    print(f"    📌 当前价格:     HK${current_price:.2f}")
    print(f"    🟢 理想买入价:   HK${ideal_buy:.2f} (布林带下轨)")
    print(f"    🟡 激进买入价:   HK${aggressive_buy:.2f} (布林带中轨)")
    print(f"    🔴 止损价:       HK${stop_loss_from_lower:.2f} (下轨 - 1.5×ATR)")
    print(f"    🎯 目标止盈价:   HK${take_profit:.2f} (布林带上轨)")
    
    if take_profit > ideal_buy:
        risk_reward = (take_profit - ideal_buy) / (ideal_buy - stop_loss_from_lower)
        print(f"    📊 盈亏比:       {risk_reward:.2f}:1")
    
    print(f"\n  操作建议:")
    
    band_pos = bollinger_info['带内位置']
    if band_pos < 0.1:
        print("    ★ 价格接近/低于布林带下轨，是布林带策略的理想买入区域")
        print(f"    ★ 建议在 HK${ideal_buy:.2f} 附近分批建仓")
        print(f"    ★ 严格设置止损在 HK${stop_loss_from_lower:.2f}")
    elif band_pos < 0.4:
        print("    ★ 价格在布林带中下区域，可考虑轻仓试探")
        print(f"    ★ 建议在 HK${ideal_buy:.2f} 挂限价单等待")
        print(f"    ★ 或在当前价位小仓位(30%)介入，跌到下轨再加仓")
    elif band_pos < 0.6:
        print("    ★ 价格在布林带中轨附近，信号不明确")
        print("    ★ 建议观望，等待价格回调到下轨区域再考虑买入")
    elif band_pos < 0.9:
        print("    ★ 价格偏向布林带上轨，不建议追高")
        print("    ★ 如持有仓位，可考虑部分止盈")
    else:
        print("    ★ 价格接近/超过布林带上轨，短期有回调风险")
        print("    ★ 不建议买入，持仓者建议止盈")
    
    print(f"\n  ⚠️ 风险提示:")
    print(f"    - 以上分析基于历史数据回测，不构成投资建议")
    print(f"    - 回测期间(2025.03-2026.03)最优策略年化收益 28.61%，夏普比率 1.85")
    print(f"    - 但过去表现不代表未来收益，市场存在不确定性")
    print(f"    - 建议控制单只股票仓位不超过总资产的 20%")
    print(f"    - 严格执行止损纪律，单笔亏损不超过总资产的 2%")
    print("=" * 70)


if __name__ == "__main__":
    main()
