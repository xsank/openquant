"""AKShare 数据源实现

基于 akshare 库获取 A 股、港股、美股、基金等多市场行情数据。
支持本地 CSV 缓存：已缓存的数据直接读取，避免频繁网络请求。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

from openquant.core.exceptions import DataSourceError
from openquant.core.interfaces import DataSourceInterface
from openquant.core.models import FrequencyType, MarketType
from openquant.datasource.retry import RetryExhaustedError, retry_fetch

logger = logging.getLogger(__name__)

# 本地缓存根目录（项目根下 data/cache/）
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
# 缓存数据必须包含的列（完整性校验）
_REQUIRED_COLUMNS = {"datetime", "open", "high", "low", "close", "volume"}


_EASTMONEY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 东方财富 push2his CDN 节点前缀池（按可用性排序，覆盖尽量多的可用节点）
_EASTMONEY_NODE_PREFIXES = [
    "80", "87", "90", "95", "75", "60", "20", "33", "40", "50",
    "55", "70", "85", "2", "3", "5", "10", "15", "25", "30", "45",
]


def _patch_requests_for_eastmoney():
    """Monkey-patch requests.get，解决东方财富 push2his 接口连接被拒绝的问题。

    根因: 当前网络环境下东方财富部分 CDN 节点(如 63)会直接关闭 TLS 连接，
    导致 AKShare 请求时触发 RemoteDisconnected 异常。

    修复方案:
    1. 为所有东方财富请求注入必要的 Referer/User-Agent headers
    2. 当节点连接失败时，自动轮转到其他 CDN 节点重试（最多尝试全部节点池）
    3. 节点之间加入 1 秒间隔，避免过于密集的请求
    """
    import time as _time
    import re as _re

    _original_get = requests.get

    # 匹配东方财富 push2his 域名
    _push2his_pattern = _re.compile(r"https?://(\d+\.)?push2his\.eastmoney\.com")

    def _patched_get(url, **kwargs):
        if not isinstance(url, str) or "eastmoney" not in url:
            return _original_get(url, **kwargs)

        # 注入 headers
        headers = kwargs.get("headers") or {}
        kwargs["headers"] = {**_EASTMONEY_HEADERS, **headers}

        # 如果不是 push2his 接口，直接请求
        if not _push2his_pattern.search(url):
            return _original_get(url, **kwargs)

        # 对 push2his 接口进行节点轮转重试
        last_error = None

        # 先尝试原始 URL
        try:
            return _original_get(url, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            logger.debug("东方财富原始节点失败，开始轮转: %s", url)

        # 原始节点失败，逐个尝试节点池中的 CDN 节点
        for prefix in _EASTMONEY_NODE_PREFIXES:
            alt_url = _push2his_pattern.sub(
                f"https://{prefix}.push2his.eastmoney.com", url
            )
            if alt_url == url:
                continue
            try:
                _time.sleep(1)
                resp = _original_get(alt_url, **kwargs)
                return resp
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                continue

        # 所有节点都失败，抛出最后的异常
        raise last_error

    requests.get = _patched_get


# 在模块加载时执行 patch
_patch_requests_for_eastmoney()


class AkshareDataSource(DataSourceInterface):
    """AKShare 数据源，支持 A 股、港股、美股、基金等多市场

    内置本地 CSV 缓存：首次获取数据后缓存到 data/cache/ 目录，
    后续请求如果缓存已覆盖所需时间范围则直接返回本地数据，避免频繁网络请求。
    """

    def get_name(self) -> str:
        return "akshare"

    def get_supported_markets(self) -> list[MarketType]:
        return [MarketType.A_SHARE, MarketType.HK_STOCK, MarketType.US_STOCK, MarketType.FUND]

    def fetch_daily_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        # 尝试从本地缓存读取
        cached_df = _load_from_cache(symbol, start_date, end_date, market)
        if cached_df is not None:
            logger.debug("缓存命中: %s (%s ~ %s)", symbol, start_date, end_date)
            return cached_df

        # 缓存未命中，计算增量获取范围（避免只差几天却重新获取全量）
        fetch_ranges = _calc_incremental_ranges(symbol, start_date, end_date, market)
        has_existing_cache = _get_cache_path(symbol, market).exists()

        try:
            fetcher = _DAILY_FETCHERS.get(market)
            if fetcher is None:
                raise DataSourceError(f"AKShare 不支持市场类型: {market}")

            incremental_frames = []
            for fetch_start, fetch_end in fetch_ranges:
                logger.debug("增量获取: %s (%s ~ %s)", symbol, fetch_start, fetch_end)
                try:
                    df_part = retry_fetch(
                        fetcher, symbol, fetch_start, fetch_end,
                        symbol=symbol,
                        retryable_exceptions=(
                            DataSourceError,
                            ConnectionError,
                            requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout,
                        ),
                    )
                    if df_part is not None and not df_part.empty:
                        incremental_frames.append(df_part)
                except RetryExhaustedError:
                    logger.warning(
                        "增量获取重试耗尽，跳过: %s (%s ~ %s)",
                        symbol, fetch_start, fetch_end,
                    )
                    continue
                except DataSourceError:
                    logger.debug("增量段无数据，跳过: %s (%s ~ %s)", symbol, fetch_start, fetch_end)
                    continue

            if not incremental_frames and not has_existing_cache:
                # 完全无缓存且网络也获取不到数据
                raise DataSourceError(f"获取数据失败: {symbol} ({start_date} ~ {end_date})")

            # 增量获取全部失败且有缓存：检查缓存是否足够新
            if not incremental_frames and has_existing_cache:
                cache_end = _get_cache_end_date(symbol, market)
                request_end_ts = pd.Timestamp(end_date)
                if cache_end is not None and (request_end_ts - cache_end).days > 1:
                    # 缓存数据过期超过1个交易日，抛异常让 multi_source 尝试备用源增量补充
                    raise DataSourceError(
                        f"AKShare 增量获取失败且缓存过期: {symbol} "
                        f"(缓存截止{cache_end.date()}, 请求截止{end_date})"
                    )

            # 合并增量数据到缓存
            if incremental_frames:
                incremental_df = pd.concat(incremental_frames, ignore_index=True)
                _save_to_cache(symbol, incremental_df, market)

        except DataSourceError:
            # 不降级，直接抛出让 multi_source 处理
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取日K线失败: {exc}") from exc

        # 从更新后的缓存中读取完整请求范围的数据
        result = _load_from_cache(symbol, start_date, end_date, market)
        if result is not None:
            return result

        # 严格模式未命中，尝试宽松模式（容忍末尾几天缺失）
        relaxed = _load_from_cache_relaxed(symbol, start_date, end_date, market)
        if relaxed is not None:
            return relaxed

        # 兜底：返回增量数据本身
        if incremental_frames:
            combined = pd.concat(incremental_frames, ignore_index=True)
            combined = combined.drop_duplicates(subset=["datetime"], keep="last")
            combined = combined.sort_values("datetime").reset_index(drop=True)
            request_start = pd.Timestamp(start_date)
            request_end = pd.Timestamp(end_date)
            mask = (combined["datetime"] >= request_start) & (combined["datetime"] <= request_end)
            return combined[mask].reset_index(drop=True)

        raise DataSourceError(f"获取数据失败: {symbol} ({start_date} ~ {end_date})")

    def fetch_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: FrequencyType = FrequencyType.MINUTE_5,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        try:
            period_map = {
                FrequencyType.MINUTE_1: "1",
                FrequencyType.MINUTE_5: "5",
                FrequencyType.MINUTE_15: "15",
                FrequencyType.MINUTE_30: "30",
                FrequencyType.MINUTE_60: "60",
            }
            period = period_map.get(frequency)
            if period is None:
                raise DataSourceError(f"AKShare 不支持分钟频率: {frequency}")

            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            return _normalize_columns(df)
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取分钟K线失败: {exc}") from exc

    def fetch_stock_list(
        self,
        market: MarketType = MarketType.A_SHARE,
    ) -> pd.DataFrame:
        try:
            if market == MarketType.A_SHARE:
                df = ak.stock_zh_a_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.A_SHARE.value
                return df[["symbol", "name", "market"]]
            elif market == MarketType.HK_STOCK:
                df = ak.stock_hk_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.HK_STOCK.value
                return df[["symbol", "name", "market"]]
            elif market == MarketType.US_STOCK:
                df = ak.stock_us_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.US_STOCK.value
                return df[["symbol", "name", "market"]]
            elif market == MarketType.FUND:
                df = ak.fund_etf_spot_em()
                df = df.rename(columns={"代码": "symbol", "名称": "name"})
                df["market"] = MarketType.FUND.value
                return df[["symbol", "name", "market"]]
            else:
                raise DataSourceError(f"AKShare 不支持市场类型: {market}")
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取股票列表失败: {exc}") from exc

    def fetch_realtime_quote(
        self,
        symbol: str,
        market: MarketType = MarketType.A_SHARE,
    ) -> dict:
        try:
            if market == MarketType.A_SHARE:
                df = ak.stock_zh_a_spot_em()
                row = df[df["代码"] == symbol]
                if row.empty:
                    raise DataSourceError(f"未找到股票: {symbol}")
                row = row.iloc[0]
                return {
                    "symbol": symbol,
                    "name": row.get("名称", ""),
                    "price": float(row.get("最新价", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                    "volume": float(row.get("成交量", 0)),
                    "amount": float(row.get("成交额", 0)),
                    "high": float(row.get("最高", 0)),
                    "low": float(row.get("最低", 0)),
                    "open": float(row.get("今开", 0)),
                    "prev_close": float(row.get("昨收", 0)),
                }
            elif market == MarketType.HK_STOCK:
                from datetime import datetime as dt, timedelta
                end = dt.now().strftime("%Y%m%d")
                start = (dt.now() - timedelta(days=7)).strftime("%Y%m%d")
                df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
                if df.empty:
                    raise DataSourceError(f"未获取到港股行情: {symbol}")
                row = df.iloc[-1]
                return {
                    "symbol": symbol,
                    "name": "",
                    "price": float(row.get("收盘", 0)),
                    "change_pct": float(row.get("涨跌幅", 0)),
                    "volume": float(row.get("成交量", 0)),
                    "amount": float(row.get("成交额", 0)),
                    "high": float(row.get("最高", 0)),
                    "low": float(row.get("最低", 0)),
                    "open": float(row.get("开盘", 0)),
                    "prev_close": float(row.get("收盘", 0)),
                }
            else:
                raise DataSourceError(f"实时行情暂不支持市场: {market}")
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"AKShare 获取实时行情失败: {exc}") from exc


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为标准格式"""
    column_mapping = {
        "日期": "datetime", "时间": "datetime",
        "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
        "成交量": "volume", "成交额": "amount", "换手率": "turnover_rate",
        "date": "datetime",
    }
    df = df.rename(columns=column_mapping)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _fetch_a_share_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到A股数据，请检查股票代码: {symbol}")
    return _normalize_columns(df)


def _fetch_hk_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.stock_hk_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到港股数据，请检查股票代码: {symbol}")
    return _normalize_columns(df)


def _resolve_us_symbol(symbol: str) -> str:
    """将用户输入的美股代码转换为 AKShare 所需的带交易所前缀格式。

    AKShare 的 stock_us_hist 接口要求 symbol 格式为 "105.GOOGL"（交易所编号.股票代码），
    东方财富使用的交易所编号：105=NASDAQ, 106=NYSE, 107=AMEX。

    解析策略（按优先级）：
    1. 用户已传入带前缀的格式（如 "105.GOOGL"）→ 直接使用
    2. 通过东方财富搜索接口精确匹配股票代码 → 获取正确的 QuoteID
    3. 搜索接口不可用时，依次尝试三大交易所前缀（105/106/107）
    """
    if "." in symbol:
        return symbol

    upper_symbol = symbol.upper()

    quote_id = _search_us_quote_id(upper_symbol)
    if quote_id:
        return quote_id

    logger.warning("无法通过搜索接口解析美股代码 %s，将依次尝试各交易所前缀", upper_symbol)
    for exchange_prefix in _US_EXCHANGE_PREFIXES:
        candidate = f"{exchange_prefix}.{upper_symbol}"
        try:
            df = ak.stock_us_hist(symbol=candidate, period="daily", adjust="qfq")
            if df is not None and not df.empty:
                logger.info("通过尝试确认美股代码: %s", candidate)
                return candidate
        except Exception:
            continue

    raise DataSourceError(
        f"无法解析美股代码: {symbol}，请使用带交易所前缀的格式（如 105.{upper_symbol}）"
    )


# 东方财富交易所编号: 105=NASDAQ, 106=NYSE, 107=AMEX
_US_EXCHANGE_PREFIXES = ("105", "106", "107")


def _search_us_quote_id(symbol: str) -> str | None:
    """通过东方财富搜索接口查找美股的完整 QuoteID。

    该接口轻量快速，无需拉取全量美股列表，返回精确匹配的 QuoteID（如 "105.GOOGL"）。
    """
    import requests

    search_url = "https://searchapi.eastmoney.com/api/suggest/get"
    params = {
        "input": symbol,
        "type": "14",
        "token": "D43BF722C8E33BDC906FB84D85E326E8",
        "count": "10",
    }
    try:
        response = requests.get(search_url, params=params, timeout=10)
        data = response.json()
        suggestions = data.get("QuotationCodeTable", {}).get("Data") or []
        for item in suggestions:
            if item.get("Code", "").upper() == symbol and item.get("SecurityTypeName") == "美股":
                return item["QuoteID"]
    except Exception:
        logger.debug("东方财富搜索接口调用失败，symbol=%s", symbol)
    return None


def _fetch_us_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    full_symbol = _resolve_us_symbol(symbol)
    df = ak.stock_us_hist(
        symbol=full_symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到美股数据，请检查股票代码: {symbol} (解析为 {full_symbol})")
    return _normalize_columns(df)


def _fetch_fund_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise DataSourceError(f"未获取到基金数据，请检查基金代码: {symbol}")
    return _normalize_columns(df)


_DAILY_FETCHERS = {
    MarketType.A_SHARE: _fetch_a_share_daily,
    MarketType.HK_STOCK: _fetch_hk_daily,
    MarketType.US_STOCK: _fetch_us_daily,
    MarketType.FUND: _fetch_fund_daily,
}


# ========== 本地 CSV 缓存 ==========


def _get_cache_path(symbol: str, market: MarketType) -> Path:
    """获取缓存文件路径: data/cache/{market}/{symbol}.csv"""
    safe_symbol = symbol.replace(".", "_").replace("/", "_")
    return _CACHE_DIR / market.value / f"{safe_symbol}.csv"


def _get_cache_end_date(symbol: str, market: MarketType) -> pd.Timestamp | None:
    """获取缓存数据的最后日期，用于判断缓存是否过期"""
    cache_path = _get_cache_path(symbol, market)
    if not cache_path.exists():
        return None
    try:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
        if df.empty:
            return None
        return df["datetime"].max()
    except Exception:
        return None


def _calc_incremental_ranges(
    symbol: str, start_date: str, end_date: str, market: MarketType
) -> list[tuple[str, str]]:
    """计算需要增量获取的时间范围。

    对比请求范围与已有缓存范围，只返回缓存未覆盖的部分：
    - 如果无缓存：返回完整请求范围
    - 如果缓存只覆盖中间段：返回前段 + 后段的增量
    - 如果缓存只缺末尾几天：只返回末尾增量（从缓存最后日期+1天开始）
    - 如果缓存只缺前面：只返回前段增量
    """
    cache_path = _get_cache_path(symbol, market)
    if not cache_path.exists():
        return [(start_date, end_date)]

    try:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
    except Exception:
        return [(start_date, end_date)]

    if df.empty or not _REQUIRED_COLUMNS.issubset(set(df.columns)):
        return [(start_date, end_date)]

    cache_start = df["datetime"].min()
    cache_end = df["datetime"].max()
    request_start = pd.Timestamp(start_date)
    request_end = pd.Timestamp(end_date)

    ranges = []

    # 前段：请求起始 早于 缓存起始（不使用容差，精确判断）
    if request_start < cache_start - pd.Timedelta(days=1):
        front_end = (cache_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ranges.append((start_date, front_end))

    # 后段：请求结束 晚于 缓存结束（只要缓存末尾 < 请求末尾就获取增量）
    if request_end > cache_end:
        back_start = (cache_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        ranges.append((back_start, end_date))

    # 如果没有增量范围但 _load_from_cache 仍判定为未命中，
    # 说明缓存覆盖但可能数据内容有问题，兜底获取全量
    if not ranges:
        return [(start_date, end_date)]

    logger.info(
        "增量获取 %s: 缓存[%s~%s], 请求[%s~%s], 增量段=%s",
        symbol, cache_start.date(), cache_end.date(), start_date, end_date, ranges,
    )
    return ranges


def _load_from_cache(
    symbol: str, start_date: str, end_date: str, market: MarketType
) -> pd.DataFrame | None:
    """从本地缓存加载数据。

    仅当缓存数据完全覆盖请求的时间范围时才返回，否则返回 None 触发网络获取。
    """
    cache_path = _get_cache_path(symbol, market)
    if not cache_path.exists():
        return None

    try:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
    except Exception as exc:
        logger.debug("读取缓存失败 %s: %s", cache_path, exc)
        return None

    # 完整性校验：必须包含必要列且数据非空
    if df.empty or not _REQUIRED_COLUMNS.issubset(set(df.columns)):
        logger.warning("缓存数据不完整，删除: %s", cache_path)
        cache_path.unlink(missing_ok=True)
        return None

    # 时间范围校验：缓存必须覆盖请求的 start_date ~ end_date
    request_start = pd.Timestamp(start_date)
    request_end = pd.Timestamp(end_date)
    cache_start = df["datetime"].min()
    cache_end = df["datetime"].max()

    # 缓存的起始日期必须 <= 请求起始日期（允许5天容差，应对节假日）
    # 缓存的结束日期必须 >= 请求结束日期 - 3天（应对最新数据延迟）
    start_covered = cache_start <= request_start + pd.Timedelta(days=5)
    end_covered = cache_end >= request_end - pd.Timedelta(days=3)

    if not (start_covered and end_covered):
        logger.debug(
            "缓存范围不足 %s: 缓存[%s~%s], 请求[%s~%s]",
            symbol, cache_start.date(), cache_end.date(), start_date, end_date,
        )
        return None

    # 按请求范围过滤返回
    mask = (df["datetime"] >= request_start) & (df["datetime"] <= request_end)
    filtered = df[mask].reset_index(drop=True)

    # 过滤后不能为空（可能日期范围内确实无交易日，但正常股票不应如此）
    if filtered.empty:
        return None

    return filtered


def _load_from_cache_relaxed(
    symbol: str, start_date: str, end_date: str, market: MarketType
) -> pd.DataFrame | None:
    """宽松模式从缓存加载数据。

    与 _load_from_cache 不同，此函数容忍末尾缺失（如周末/节假日），
    只要缓存起始覆盖且缓存中有足够的数据就返回。
    用于增量获取失败时的降级回退。
    """
    cache_path = _get_cache_path(symbol, market)
    if not cache_path.exists():
        return None

    try:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
    except Exception:
        return None

    if df.empty or not _REQUIRED_COLUMNS.issubset(set(df.columns)):
        return None

    request_start = pd.Timestamp(start_date)
    request_end = pd.Timestamp(end_date)
    cache_start = df["datetime"].min()

    # 宽松模式：只要求缓存起始覆盖请求起始（5天容差）
    if cache_start > request_start + pd.Timedelta(days=5):
        return None

    # 按请求范围过滤（末尾可能不到 request_end，但有数据就返回）
    mask = (df["datetime"] >= request_start) & (df["datetime"] <= request_end)
    filtered = df[mask].reset_index(drop=True)

    if filtered.empty:
        return None

    return filtered


def _save_to_cache(symbol: str, df: pd.DataFrame, market: MarketType) -> None:
    """将数据保存到本地缓存。

    缓存策略：与已有缓存合并（取并集），保留最大范围的数据。
    安全措施：只缓存非空且列完整的 DataFrame，避免缓存空数据导致回测错误。
    """
    if df is None or df.empty:
        logger.debug("数据为空，不缓存: %s", symbol)
        return

    if not _REQUIRED_COLUMNS.issubset(set(df.columns)):
        logger.debug("数据列不完整，不缓存: %s (列: %s)", symbol, list(df.columns))
        return

    cache_path = _get_cache_path(symbol, market)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 确保 datetime 列是 Timestamp 类型
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 如果已有缓存，合并数据（取并集，去重保留最新）
    if cache_path.exists():
        try:
            existing = pd.read_csv(cache_path, parse_dates=["datetime"])
            if not existing.empty and _REQUIRED_COLUMNS.issubset(set(existing.columns)):
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["datetime"], keep="last")
                combined = combined.sort_values("datetime").reset_index(drop=True)
                df = combined
        except Exception as exc:
            logger.debug("合并缓存失败 %s: %s，覆盖写入", cache_path, exc)

    # 最终校验：合并后数据必须非空
    if df.empty:
        return

    df.to_csv(cache_path, index=False)
    logger.debug("缓存已更新: %s (%d 条数据)", cache_path, len(df))
