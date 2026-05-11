#!/bin/bash
# OpenQuant 股票买入/卖出推荐脚本
# 基于多策略 × 滚动窗口验证（Walk-Forward Analysis）
# 训练窗口: 20周, 滚动验证: 15轮, 总数据需求: 约35周

cd "$(dirname "$0")"

echo "============================================================"
echo "  OpenQuant 股票买入/卖出推荐（滚动验证版）"
echo "  标的: 19只 (港股3 + A股2 + 美股14)"
echo "  策略: 9种综合分析"
echo "  验证方式: 15轮滚动窗口 × 20周训练"
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
    us_stock:106.BABA:阿里巴巴 \
    us_stock:105.FUTU:富途控股 \
    us_stock:105.MU:美光科技 \
    us_stock:106.NEM:纽蒙特矿业 \
    us_stock:106.IONQ:IonQ \
    us_stock:106.YANG:中国3倍做空ETF \
    us_stock:106.MP:MP_Materials \
  --train-weeks 20 \
  --rolling-rounds 15 \
  --datasource multi_source \
  --capital 100000

echo ""
echo "分析完成！"
