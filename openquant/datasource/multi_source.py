"""多源聚合数据源

按优先级尝试多个数据源获取数据：akshare(东方财富) → yfinance → ...
每个数据源内部自带重试机制，某数据源全部重试耗尽后降级到下一个。
成功获取的数据会缓存到本地，格式与 akshare 缓存完全一致，可互通。
"""
from __future__ import annotations

import logging

import pandas as pd

from openquant.core.exceptions import DataSourceError
from openquant.core.interfaces import DataSourceInterface
from openquant.core.models import FrequencyType, MarketType
from openquant.datasource.retry import RetryExhaustedError, retry_fetch

logger = logging.getLogger(__name__)

# 必须包含的列（完整性校验）
_REQUIRED_COLUMNS = {"datetime", "open", "high", "low", "close", "volume"}


class MultiSourceDataSource(DataSourceInterface):
    """多源聚合数据源

    按优先级依次尝试多个数据源，某个源失败后自动降级到下一个。
    成功获取的数据通过 akshare 的缓存机制本地持久化。

    降级链路: akshare → yfinance
    - akshare 覆盖 A股/港股/美股/基金
    - yfinance 覆盖 美股/港股/A股（作为备用）
    """

    def __init__(self):
        self._sources: list[DataSourceInterface] = []
        self._initialized = False

    def _ensure_sources(self) -> None:
        """延迟初始化数据源实例，避免循环导入"""
        if self._initialized:
            return
        self._initialized = True

        from openquant.datasource.akshare_source import AkshareDataSource
        from openquant.datasource.yfinance_source import YFinanceDataSource

        self._sources = [
            AkshareDataSource(),
            YFinanceDataSource(),
        ]
        logger.info(
            "多源数据源初始化完成，降级链路: %s",
            " → ".join(s.get_name() for s in self._sources),
        )

    def get_name(self) -> str:
        return "multi_source"

    def get_supported_markets(self) -> list[MarketType]:
        return [
            MarketType.A_SHARE,
            MarketType.HK_STOCK,
            MarketType.US_STOCK,
            MarketType.FUND,
        ]

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        self._ensure_sources()

        errors: list[str] = []
        for source in self._sources:
            if market not in source.get_supported_markets():
                continue

            source_name = source.get_name()
            try:
                logger.info(
                    "[%s] 尝试通过 %s 获取数据: %s (%s ~ %s)",
                    symbol, source_name, symbol, start_date, end_date,
                )
                df = source.fetch_daily_bars(symbol, start_date, end_date, market)

                if df is not None and not df.empty:
                    # 完整性校验
                    missing = _REQUIRED_COLUMNS - set(df.columns)
                    if missing:
                        logger.warning(
                            "[%s] %s 返回数据缺少列: %s，跳过",
                            symbol, source_name, missing,
                        )
                        errors.append(f"{source_name}: 数据缺少列 {missing}")
                        continue

                    # 如果不是 akshare（主源已自带缓存），手动写入缓存
                    if source_name != "akshare":
                        self._save_fallback_to_cache(symbol, df, market)

                    logger.info(
                        "[%s] 通过 %s 成功获取 %d 条数据",
                        symbol, source_name, len(df),
                    )
                    return df

                errors.append(f"{source_name}: 返回空数据")

            except (DataSourceError, RetryExhaustedError) as exc:
                error_msg = f"{source_name}: {exc}"
                errors.append(error_msg)
                logger.warning(
                    "[%s] %s 获取失败: %s，尝试下一个数据源",
                    symbol, source_name, exc,
                )
                continue
            except Exception as exc:
                error_msg = f"{source_name}: {type(exc).__name__}: {exc}"
                errors.append(error_msg)
                logger.warning(
                    "[%s] %s 获取异常: %s，尝试下一个数据源",
                    symbol, source_name, exc,
                )
                continue

        # 所有数据源均失败
        all_errors = "; ".join(errors)
        raise DataSourceError(
            f"所有数据源获取 {symbol} 均失败 ({start_date}~{end_date}): {all_errors}"
        )

    @staticmethod
    def _save_fallback_to_cache(
        symbol: str, df: pd.DataFrame, market: MarketType
    ) -> None:
        """将备用数据源获取的数据写入 akshare 的缓存目录，保持格式统一"""
        try:
            from openquant.datasource.akshare_source import _save_to_cache
            _save_to_cache(symbol, df, market)
            logger.info("[%s] 备用源数据已写入本地缓存", symbol)
        except Exception as exc:
            logger.warning("[%s] 备用源数据写入缓存失败: %s", symbol, exc)

    def fetch_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: FrequencyType = FrequencyType.MINUTE_5,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        self._ensure_sources()
        # 分钟K线只有 akshare 支持，直接代理
        for source in self._sources:
            if market not in source.get_supported_markets():
                continue
            try:
                return source.fetch_minute_bars(
                    symbol, start_date, end_date, frequency, market
                )
            except (DataSourceError, NotImplementedError):
                continue
        raise DataSourceError(f"所有数据源均不支持 {market} 的分钟K线获取")

    def fetch_stock_list(
        self,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        self._ensure_sources()
        for source in self._sources:
            if market not in source.get_supported_markets():
                continue
            try:
                return source.fetch_stock_list(market)
            except DataSourceError:
                continue
        raise DataSourceError(f"所有数据源均不支持 {market} 的股票列表获取")

    def fetch_realtime_quote(
        self,
        symbol: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> dict:
        self._ensure_sources()
        errors: list[str] = []
        for source in self._sources:
            if market not in source.get_supported_markets():
                continue
            try:
                return source.fetch_realtime_quote(symbol, market)
            except DataSourceError as exc:
                errors.append(f"{source.get_name()}: {exc}")
                continue
        raise DataSourceError(
            f"所有数据源获取 {symbol} 实时行情均失败: {'; '.join(errors)}"
        )
