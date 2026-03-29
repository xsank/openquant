"""SQLite 存储实现"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from openquant.core.exceptions import StorageError
from openquant.core.interfaces import StorageInterface
from openquant.core.models import (
    FrequencyType,
    MarketType,
    OrderSide,
    Portfolio,
    TradeRecord,
)


class SqliteStorage(StorageInterface):
    """基于 SQLite 的本地轻量级存储"""

    def __init__(self, db_path: str = "openquant.db"):
        self.db_path = db_path
        self._connection: sqlite3.Connection | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
        return self._connection

    def initialize(self) -> None:
        cursor = self.connection.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                amount REAL DEFAULT 0,
                turnover_rate REAL DEFAULT 0,
                market TEXT NOT NULL,
                frequency TEXT NOT NULL DEFAULT 'daily',
                PRIMARY KEY (symbol, datetime, market, frequency)
            );

            CREATE INDEX IF NOT EXISTS idx_bars_symbol_date
                ON bars(symbol, datetime);

            CREATE TABLE IF NOT EXISTS trade_records (
                trade_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                commission REAL DEFAULT 0,
                traded_at TEXT NOT NULL,
                market TEXT NOT NULL DEFAULT 'a_share'
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time TEXT NOT NULL,
                cash REAL NOT NULL,
                total_equity REAL NOT NULL,
                total_market_value REAL NOT NULL,
                positions_json TEXT,
                initial_capital REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS stock_list (
                symbol TEXT NOT NULL,
                name TEXT,
                market TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (symbol, market)
            );
        """)
        self.connection.commit()

    def save_bars(self, bars: pd.DataFrame, symbol: str, market: MarketType) -> int:
        if bars.empty:
            return 0
        required_columns = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required_columns - set(bars.columns)
        if missing:
            raise StorageError(f"DataFrame 缺少必要列: {missing}")

        records = []
        for _, row in bars.iterrows():
            records.append((
                symbol,
                str(row["datetime"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                float(row.get("amount", 0)),
                float(row.get("turnover_rate", 0)),
                market.value,
                row.get("frequency", FrequencyType.DAILY.value),
            ))

        self.connection.executemany(
            "INSERT OR REPLACE INTO bars "
            "(symbol, datetime, open, high, low, close, volume, amount, turnover_rate, market, frequency) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        self.connection.commit()
        return len(records)

    def load_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        market: MarketType = MarketType.A_SHARE,
        frequency: FrequencyType = FrequencyType.DAILY,
    ) -> pd.DataFrame:
        query = (
            "SELECT datetime, open, high, low, close, volume, amount, turnover_rate "
            "FROM bars WHERE symbol = ? AND datetime >= ? AND datetime <= ? "
            "AND market = ? AND frequency = ? ORDER BY datetime"
        )
        df = pd.read_sql_query(
            query,
            self.connection,
            params=(symbol, start_date, end_date, market.value, frequency.value),
        )
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def save_trade_records(self, records: list[TradeRecord]) -> int:
        if not records:
            return 0
        data = [
            (
                record.trade_id,
                record.order_id,
                record.symbol,
                record.side.value,
                record.price,
                record.quantity,
                record.commission,
                record.traded_at.isoformat(),
                record.market.value,
            )
            for record in records
        ]
        self.connection.executemany(
            "INSERT OR REPLACE INTO trade_records "
            "(trade_id, order_id, symbol, side, price, quantity, commission, traded_at, market) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            data,
        )
        self.connection.commit()
        return len(data)

    def load_trade_records(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[TradeRecord]:
        query = "SELECT * FROM trade_records WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND traded_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND traded_at <= ?"
            params.append(end_date)
        query += " ORDER BY traded_at"

        cursor = self.connection.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append(TradeRecord(
                trade_id=row["trade_id"],
                order_id=row["order_id"],
                symbol=row["symbol"],
                side=OrderSide(row["side"]),
                price=row["price"],
                quantity=row["quantity"],
                commission=row["commission"],
                traded_at=datetime.fromisoformat(row["traded_at"]),
                market=MarketType(row["market"]),
            ))
        return results

    def save_portfolio_snapshot(self, portfolio: Portfolio, snapshot_time: datetime) -> None:
        positions_data = {
            symbol: {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market": pos.market.value,
            }
            for symbol, pos in portfolio.positions.items()
        }
        self.connection.execute(
            "INSERT INTO portfolio_snapshots "
            "(snapshot_time, cash, total_equity, total_market_value, positions_json, initial_capital) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                snapshot_time.isoformat(),
                portfolio.cash,
                portfolio.total_equity,
                portfolio.total_market_value,
                json.dumps(positions_data, ensure_ascii=False),
                portfolio.initial_capital,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
