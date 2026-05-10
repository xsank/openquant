#!/bin/bash
# OpenQuant 回测验证脚本
# 用历史数据验证不同概率阈值下的策略收益
# 找到最优阈值组合，确保高置信度时收益最大化

cd "$(dirname "$0")"

echo "============================================================"
echo "  OpenQuant 回测验证 - 概率阈值优化"
echo "  标的: 3只 (快速验证)"
echo "  验证窗口: 10周"
echo "  阈值范围: 买入30-70%, 卖出30-70%"
echo "============================================================"

python -m openquant.main validate \
  --stocks \
    us_stock:105.AMD:AMD \
    hk_stock:00700:腾讯 \
    us_stock:105.GOOG:谷歌 \
  --train-weeks 15 \
  --rolling-rounds 8 \
  --validation-weeks 10 \
  --buy-thresholds "30,40,50,60,70" \
  --sell-thresholds "30,40,50,60,70" \
  --datasource akshare \
  --capital 100000

echo ""
echo "验证完成！"
