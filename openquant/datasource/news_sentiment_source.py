"""新闻情绪数据源

通过 akshare 获取股票相关新闻，并进行简单的情绪分析。
支持基于关键词的正负向情绪判断，为情绪分析模块提供新闻维度的数据。

情绪判断规则：
- 正向关键词（利多）：业绩增长、回购、增持、突破、超预期等
- 负向关键词（利空）：亏损、减持、违规、诉讼、下调等
- 中性：无明显情绪倾向的新闻
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# 正向情绪关键词及权重
_BULLISH_KEYWORDS: list[tuple[str, float]] = [
    ("业绩增长", 1.0),
    ("净利润增长", 1.0),
    ("营收增长", 0.8),
    ("超预期", 1.2),
    ("业绩超预期", 1.5),
    ("回购", 0.8),
    ("增持", 0.9),
    ("大股东增持", 1.2),
    ("管理层增持", 1.0),
    ("战略合作", 0.6),
    ("重大合同", 0.9),
    ("中标", 0.8),
    ("突破", 0.5),
    ("创新高", 0.7),
    ("分红", 0.6),
    ("派息", 0.6),
    ("上调评级", 1.0),
    ("买入评级", 0.9),
    ("强烈推荐", 0.8),
    ("利好", 0.7),
    ("政策支持", 0.7),
    ("获批", 0.8),
    ("上市", 0.6),
    ("并购", 0.5),
    ("扭亏为盈", 1.2),
    ("盈利", 0.6),
    ("盈利增长", 0.9),
]

# 负向情绪关键词及权重
_BEARISH_KEYWORDS: list[tuple[str, float]] = [
    ("亏损", 1.0),
    ("净利润下滑", 1.0),
    ("营收下滑", 0.8),
    ("业绩下滑", 1.0),
    ("业绩不及预期", 1.2),
    ("减持", 0.9),
    ("大股东减持", 1.2),
    ("清仓", 1.3),
    ("违规", 1.0),
    ("处罚", 1.0),
    ("立案调查", 1.5),
    ("诉讼", 0.8),
    ("债务违约", 1.5),
    ("流动性危机", 1.5),
    ("退市", 2.0),
    ("ST", 1.0),
    ("下调评级", 1.0),
    ("卖出评级", 0.9),
    ("减持评级", 0.8),
    ("利空", 0.7),
    ("监管", 0.5),
    ("风险", 0.4),
    ("亏损扩大", 1.2),
    ("商誉减值", 1.0),
    ("计提减值", 0.9),
    ("业绩预警", 1.1),
    ("暂停", 0.7),
    ("停产", 0.8),
]


class NewsSentimentSource:
    """新闻情绪数据源

    通过 akshare 获取个股新闻，并基于关键词进行情绪打分。
    结果以标准化的 dict 列表返回，供 SentimentAnalyzer 使用。
    """

    def get_name(self) -> str:
        return "news_sentiment"

    def fetch_news_sentiment(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        max_items: int = 50,
    ) -> list[dict]:
        """获取个股新闻并进行情绪分析

        Args:
            symbol: 股票代码（如 '000001' 或港股 '09988'）
            start_date: 开始日期（'YYYY-MM-DD'，仅作软过滤参考）
            end_date: 结束日期（'YYYY-MM-DD'，仅作软过滤参考）
            max_items: 最多返回的新闻条数

        Returns:
            新闻情绪列表，每条包含：
            - datetime: 新闻时间
            - title: 新闻标题
            - sentiment: 'positive' / 'negative' / 'neutral'
            - score: 情绪得分（正为利多，负为利空，范围约 [-2, 2]）
            - source: 数据来源

        注意：
            akshare 的新闻接口只返回最近约 10 条，无法按历史日期查询。
            本方法不严格过滤日期，直接返回所有能获取到的新闻，
            由 SentimentAnalyzer 的时间衰减机制自然处理历史权重。
        """
        news_items: list[dict] = []

        try:
            import akshare as ak
            raw_news = self._fetch_from_akshare(ak, symbol, start_date, end_date)
        except ImportError:
            logger.warning("akshare 未安装，新闻情绪数据不可用")
            return []
        except Exception as exc:
            logger.warning("获取 %s 新闻数据失败: %s", symbol, exc)
            return []

        end_ts = pd.Timestamp(end_date)
        # 软过滤：只丢弃明显早于 start_date 超过 180 天的新闻（避免极端过期数据）
        # 不丢弃晚于 end_date 的新闻（因为接口只能获取最新数据，时间衰减会自然降权）
        soft_start_ts = pd.Timestamp(start_date) - pd.Timedelta(days=180)

        for raw_item in raw_news[:max_items]:
            title = str(raw_item.get("title", ""))
            content = str(raw_item.get("content", ""))
            pub_time = raw_item.get("datetime")

            if pub_time is None:
                continue

            try:
                pub_ts = pd.Timestamp(pub_time)
            except (ValueError, TypeError):
                continue

            # 软过滤：只丢弃极端过期数据
            if pub_ts < soft_start_ts:
                continue

            sentiment_label, score = self._analyze_sentiment(title + " " + content)

            news_items.append({
                "datetime": pub_ts.to_pydatetime(),
                "title": title,
                "sentiment": sentiment_label,
                "score": score,
                "source": "akshare_news",
                "symbol": symbol,
            })

        logger.info("获取 %s 的 %d 条新闻情绪数据（接口返回最近新闻，时间衰减自动降权）", symbol, len(news_items))
        return news_items

    def fetch_latest_news_sentiment(
        self,
        symbol: str,
        limit: int = 20,
    ) -> list[dict]:
        """获取最新新闻情绪（用于实时交易）"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        return self.fetch_news_sentiment(symbol, start_date, end_date, max_items=limit)

    def _fetch_from_akshare(
        self,
        ak,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[dict]:
        """通过 akshare 获取原始新闻数据"""
        raw_items: list[dict] = []

        # 尝试获取个股新闻
        try:
            df = ak.stock_news_em(symbol=symbol)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    pub_time = row.get("发布时间", row.get("时间", None))
                    title = str(row.get("新闻标题", row.get("标题", "")))
                    content = str(row.get("新闻内容", row.get("内容", "")))

                    raw_items.append({
                        "datetime": pub_time,
                        "title": title,
                        "content": content,
                    })
        except Exception as exc:
            logger.debug("akshare stock_news_em 获取失败: %s", exc)

        # 尝试获取财经新闻（作为补充）
        if not raw_items:
            try:
                df = ak.stock_news_main_sina(symbol=symbol)
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        pub_time = row.get("date", row.get("时间", None))
                        title = str(row.get("title", row.get("标题", "")))
                        raw_items.append({
                            "datetime": pub_time,
                            "title": title,
                            "content": "",
                        })
            except Exception as exc:
                logger.debug("akshare stock_news_main_sina 获取失败: %s", exc)

        return raw_items

    def _analyze_sentiment(self, text: str) -> tuple[str, float]:
        """基于关键词分析文本情绪

        Args:
            text: 待分析的文本（标题 + 内容）

        Returns:
            (sentiment_label, score)
            - sentiment_label: 'positive' / 'negative' / 'neutral'
            - score: 情绪得分，正为利多，负为利空
        """
        bullish_score = 0.0
        bearish_score = 0.0

        for keyword, weight in _BULLISH_KEYWORDS:
            if keyword in text:
                bullish_score += weight

        for keyword, weight in _BEARISH_KEYWORDS:
            if keyword in text:
                bearish_score += weight

        net_score = bullish_score - bearish_score

        # 归一化到 [-2, 2] 范围
        max_possible = 3.0
        normalized_score = max(-2.0, min(2.0, net_score / max_possible * 2.0))

        if normalized_score > 0.1:
            sentiment_label = "positive"
        elif normalized_score < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return sentiment_label, normalized_score
