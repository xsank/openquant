# openquant

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
  --symbol 600519 --start-date 2023-01-01 --end-date 2024-01-01 \
  --strategy ma_cross --capital 100000
```

### 2.2 Simulate
```python
python -m openquant.main simulate \
  --symbols 600519 000001 --strategy macd \
  --datasource akshare --interval 60 --max-rounds 3
```