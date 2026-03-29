#!/bin/bash
# OpenQuant 批量回测脚本
# 对港股、A股、美股的12只标的使用全部8种策略进行回测
# 回测区间: 2025-01-01 ~ 2026-01-01

cd "$(dirname "$0")"

echo "============================================================"
echo "  OpenQuant 批量回测"
echo "  标的: 12只 (港股3 + A股2 + 美股7)"
echo "  策略: 8种 (ma_cross, macd, rsi_reversal, bollinger_band,"
echo "         turtle, kdj, dual_momentum, volume_breakout)"
echo "  区间: 2025-01-01 ~ 2026-01-01"
echo "============================================================"

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
    us_stock:105.ZH:知乎 \
    us_stock:105.BILI:哔哩哔哩 \
    us_stock:105.TSM:台积电 \
    us_stock:105.AMD:AMD \
  --start-date 2025-01-01 \
  --end-date 2026-01-01 \
  --datasource akshare \
  --capital 100000 \
  --output-dir output/charts

echo ""
echo "回测完成！图表已保存到 output/charts 目录"
