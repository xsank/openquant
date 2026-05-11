"""通用重试与请求间隔工具

提供指数退避重试装饰器和请求限速器，供多个数据源共用。
"""
from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 默认配置
DEFAULT_MAX_RETRIES = 9
DEFAULT_BASE_DELAY = 2.0        # 首次重试等待秒数
DEFAULT_MAX_DELAY = 60.0        # 最大等待秒数
DEFAULT_BACKOFF_FACTOR = 2.0    # 指数退避因子
DEFAULT_REQUEST_INTERVAL = 1.5  # 连续请求间最小间隔秒数


class RetryExhaustedError(Exception):
    """重试次数耗尽后抛出的异常"""

    def __init__(self, symbol: str, attempts: int, last_error: Exception | None = None):
        self.symbol = symbol
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"获取 {symbol} 数据失败，已重试 {attempts} 次: {last_error}"
        )


class RequestThrottle:
    """请求限速器：确保连续请求之间有最小间隔

    线程不安全，适用于单线程顺序数据获取场景。
    """

    def __init__(self, min_interval: float = DEFAULT_REQUEST_INTERVAL):
        self._min_interval = min_interval
        self._last_request_time = 0.0

    def wait(self) -> None:
        """等待到满足最小间隔后返回"""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            sleep_time = self._min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.monotonic()

    def mark(self) -> None:
        """标记一次请求已完成（不等待）"""
        self._last_request_time = time.monotonic()


# 全局限速器实例，多数据源共享
_global_throttle = RequestThrottle()


def get_global_throttle() -> RequestThrottle:
    """获取全局请求限速器"""
    return _global_throttle


def retry_fetch(
    fetch_func: Callable[..., T],
    *args,
    symbol: str = "",
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    throttle: RequestThrottle | None = None,
    **kwargs,
) -> T:
    """带指数退避的重试执行函数

    Args:
        fetch_func: 要重试执行的函数
        *args: 传给 fetch_func 的位置参数
        symbol: 标的代码（用于日志）
        max_retries: 最大重试次数（默认9次）
        base_delay: 首次重试等待秒数
        max_delay: 单次等待上限秒数
        backoff_factor: 指数退避因子
        retryable_exceptions: 可重试的异常类型元组
        throttle: 限速器实例，None则使用全局限速器
        **kwargs: 传给 fetch_func 的关键字参数

    Returns:
        fetch_func 的返回值

    Raises:
        RetryExhaustedError: 所有重试均失败
    """
    if throttle is None:
        throttle = _global_throttle

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            throttle.wait()
            result = fetch_func(*args, **kwargs)
            throttle.mark()
            return result
        except retryable_exceptions as exc:
            last_error = exc
            throttle.mark()

            if attempt >= max_retries:
                break

            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            logger.warning(
                "[%s] 第 %d/%d 次获取失败: %s，%.1f秒后重试",
                symbol, attempt, max_retries,
                type(exc).__name__, delay,
            )
            time.sleep(delay)

    raise RetryExhaustedError(symbol, max_retries, last_error)
