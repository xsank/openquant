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