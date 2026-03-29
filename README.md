# openquant

<p align="center">
  <img src="doc/openquant.jpg" alt="OpenQuant Logo" width="200">
</p>

Personal quantitative trading system, hoping to achieve financial freedom...

## 1. Installation
```bash
git clone https://github.com/xsank/openquant.git
cd openquant
pip install -r requirements.txt
``` 

## 2. Usage

### 2.1 Backtest
```python
python -m openquant.main backtest \
  --symbol 09988 --start-date 2025-01-01 --end-date 2026-01-01 \
  --market hk_stock \
  --strategy ma_cross --capital 100000
```

the result is as follows:
```text
============================================================
  策略: MA_Cross(5,20)
============================================================
  初始资金:           100,000.00
  最终权益:           113,729.62
  总收益率:               13.73%
  年化收益率:             14.09%
  年化波动率:             26.96%
  夏普比率:               0.4113
  索提诺比率:             0.5832
  卡玛比率:               0.6894
  最大回撤:              -20.43%
  最大回撤天数:               84
  胜率:                   16.33%
  盈亏比:                 1.4003
  交易次数:                   10
  总佣金:                 263.16
============================================================
```

### 2.2 Simulate
```python
python -m openquant.main simulate \
  --symbols 09988 --strategy ma_cross \
  --datasource akshare --market hk_stock \
  --capital 100000 --interval 5 --max-rounds 3
```

Warning! you should know exactly what you are doing!

### 2.3 Batch Analysis

支持多标的 × 多策略的批量回测，自动生成对比图表：

```bash
python -m openquant.main batch_backtest \
  --stocks \
    hk_stock:01810:小米 \
    hk_stock:00700:腾讯 \
    hk_stock:03690:美团 \
    a_share:300750:宁德时代 \
    a_share:002594:比亚迪 \
    us_stock:105.GOOG:谷歌 \
    us_stock:105.PDD:拼多多 \
    us_stock:105.TCOM:携程 \
    us_stock:105.BILI:哔哩哔哩 \
    us_stock:105.AMD:AMD \
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --datasource akshare \
  --capital 100000 \
  --output-dir output/charts
```
or  
```bash
bash batch_backtest.sh
```

## 3. Backtest Results (2025-01-01 ~ 2026-01-01)

The following are 10 stocks including Hong Kong stocks (小米, 腾讯, 美团), A-shares (CATL, 比亚迪), and US stocks (谷歌, 拼多多, 携程, 哔哩哔哩, AMD). Analysis of the backtest results using eight strategies (MA Cross, MACD, RSI Reversal, Bollinger Band, Turtle, KDJ, Dual Momentum, Volume Breakout).

### 3.1 Return Heatmap

The yield heat map shows the total yield performance of each strategy on each asset. The greener the color, the higher the return; the redder the color, the greater the loss:

<p align="center">
  <img src="doc/return_heatmap.png" alt="Return Heatmap" width="800">
</p>

### 3.2 Best Strategy per Stock

Comparison of the returns of the best and worst strategies for each target:

<p align="center">
  <img src="doc/best_strategy_per_stock.png" alt="Best Strategy per Stock" width="800">
</p>

**Key Findings:**

| Stock | Best Strategy | Return |
|-------|--------------|--------|
| AMD | MA Cross | +122.20% |
| 携程 | RSI Reversal | +42.53% |
| 谷歌 | Dual Momentum | +41.74% |
| 宁德时代 | MACD | +27.19% |
| 哔哩哔哩 | KDJ | +19.38% |
| 比亚迪 | Bollinger Band | +17.25% |
| 腾讯 | Turtle | +17.15% |
| 小米 | KDJ | +16.03% |
| 拼多多 | RSI Reversal | +14.02% |
| 美团 | KDJ | +0.93% |
