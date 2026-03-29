"""日期工具函数"""
from __future__ import annotations

from datetime import datetime, timedelta


def parse_date(date_str: str) -> datetime:
    """解析日期字符串，支持多种格式"""
    formats = ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期: {date_str}")


def format_date(dt: datetime, fmt: str = "%Y-%m-%d") -> str:
    """格式化日期"""
    return dt.strftime(fmt)


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """生成日期范围内的工作日列表（简单排除周末）"""
    start = parse_date(start_date)
    end = parse_date(end_date)
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(format_date(current))
        current += timedelta(days=1)
    return days


def days_between(date1: str, date2: str) -> int:
    """计算两个日期之间的天数"""
    d1 = parse_date(date1)
    d2 = parse_date(date2)
    return abs((d2 - d1).days)
