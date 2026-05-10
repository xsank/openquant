#!/usr/bin/env python3
"""验证6只新股票在akshare中的代码是否正确"""
import akshare as ak
import time
import sys

stocks_to_verify = [
    ('105.FUTU', 'FUTU富途控股-纳斯达克'),
    ('105.MU', 'MU美光科技-纳斯达克'),
    ('106.NEM', 'NEM纽蒙特矿业-纽交所'),
    ('106.IONQ', 'IONQ-纽交所'),
    ('106.YANG', 'YANG做空ETF-纽交所'),
    ('106.MP', 'MP Materials-纽交所'),
]

results = []
for code, desc in stocks_to_verify:
    success = False
    for attempt in range(5):
        try:
            df = ak.stock_us_hist(symbol=code, period='daily', adjust='qfq')
            if df is not None and len(df) > 0:
                latest = df.iloc[-1]['日期'] if '日期' in df.columns else 'N/A'
                line = f'OK {desc} ({code}): {len(df)} rows, latest={latest}'
                results.append(line)
                print(line, flush=True)
                success = True
                break
            else:
                line = f'EMPTY {desc} ({code})'
                results.append(line)
                print(line, flush=True)
                success = True
                break
        except Exception as e:
            print(f'  attempt {attempt+1} failed for {code}: {str(e)[:60]}', flush=True)
            time.sleep(10)  # 等待10秒重试
    if not success:
        results.append(f'ERR {desc} ({code}): all attempts failed')
    time.sleep(8)  # 每只股票间隔8秒

print("\n=== SUMMARY ===")
for r in results:
    print(r)

with open('/tmp/stock_verify_result.txt', 'w') as f:
    for r in results:
        f.write(r + '\n')
print("\nResults saved to /tmp/stock_verify_result.txt")
