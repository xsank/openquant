#!/bin/bash
# OpenQuant 股票买入推荐脚本
# 基于多策略综合分析，筛选最适合今日买入的股票
# 回测区间: 近9周历史数据

cd "$(dirname "$0")"

echo "============================================================"
echo "  OpenQuant 股票买入推荐"
echo "  标的: 12只 (港股3 + A股2 + 美股7)"
echo "  策略: 8种综合分析"
echo "  回测周期: 近9周"
echo "============================================================"

python -m openquant.main recommend \
  --stocks \
    hk_stock:01810:小米 \
    hk_stock:00700:腾讯 \
    hk_stock:03690:美团 \
    a_share:300750:宁德时代 \
    a_share:002594:比亚迪 \
    us_stock:105.GOOG:谷歌 \
    us_stock:105.PDD:拼多多 \
    us_stock:105.TCOM:携程 \
    us_stock:106.ZH:知乎 \
    us_stock:105.BILI:哔哩哔哩 \
    us_stock:106.TSM:台积电 \
    us_stock:105.AMD:AMD \
  --weeks 9 \
  --datasource akshare \
  --capital 100000

echo ""
echo "分析完成！"
