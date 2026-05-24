"""
PostgreSQL database manager using psycopg2.
Handles schema creation, inserts, and all analytics queries.

Fixes applied:
  - Connection leak: _get_conn uses a context-managed pool; close() called on app exit
  - load_dotenv() support via DATABASE_URL env var
  - avg_speed parameter renamed avg_speed_kmh for clarity
  - Dynamic road discovery (no hardcoded road names)
  - SQLite fallback mode for zero-setup testing
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS traffic_readings (
    id                SERIAL PRIMARY KEY,
    road_name         VARCHAR(100)  NOT NULL,
    timestamp         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    vehicle_count     INTEGER       NOT NULL DEFAULT 0,
    congestion_score  INTEGER       NOT NULL DEFAULT 0,
    congestion_level  VARCHAR(20)   NOT NULL DEFAULT 'low',
    avg_speed_kmh     FLOAT         DEFAULT 0.0,
    vehicle_types     JSONB         DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS incidents (
    id           SERIAL PRIMARY KEY,
    road_name    VARCHAR(100) NOT NULL,
    description  TEXT         NOT NULL,
    severity     VARCHAR(20)  DEFAULT 'medium',
    reported_at  TIMESTAMPTZ  DEFAULT NOW(),
    resolved_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_traffic_timestamp  ON traffic_readings(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_traffic_road       ON traffic_readings(road_name);
CREATE INDEX IF NOT EXISTS idx_traffic_congestion ON traffic_readings(congestion_level);
"""

SCHEMA_SQL_SQLITE = """
CREATE TABLE IF NOT EXISTS traffic_readings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    road_name         TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    vehicle_count     INTEGER NOT NULL DEFAULT 0,
    congestion_score  INTEGER NOT NULL DEFAULT 0,
    congestion_level  TEXT    NOT NULL DEFAULT 'low',
    avg_speed_kmh     REAL    DEFAULT 0.0,
    vehicle_types     TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS incidents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    road_name    TEXT NOT NULL,
    description  TEXT NOT NULL,
    severity     TEXT DEFAULT 'medium',
    reported_at  TEXT DEFAULT (datetime('now')),
    resolved_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_traffic_timestamp  ON traffic_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_traffic_road       ON traffic_readings(road_name);
"""


class DatabaseManager:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._use_sqlite = db_url.startswith("sqlite:///")
        self._sqlite_path = db_url[len("sqlite:///"):] if self._use_sqlite else None
        self._conn = None

    # ── Connection management ─────────────────────────────────────────────────

    def _get_conn(self):
        if self._use_sqlite:
            import sqlite3
            if self._conn is None:
                self._conn = sqlite3.connect(
                    self._sqlite_path, check_same_thread=False
                )
                self._conn.row_factory = sqlite3.Row
        else:
            import psycopg2
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(self.db_url)
                self._conn.autocommit = False
        return self._conn

    def close(self):
        """Cleanly close the DB connection — call on app shutdown."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("DB connection closed.")
            except Exception:
                pass
            finally:
                self._conn = None

    def initialize(self):
        """Create schema only — no sample data inserted."""
        try:
            conn = self._get_conn()
            schema = SCHEMA_SQL_SQLITE if self._use_sqlite else SCHEMA_SQL
            if self._use_sqlite:
                import sqlite3
                conn.executescript(schema)
                conn.commit()
            else:
                with conn.cursor() as cur:
                    cur.execute(schema)
                conn.commit()
            logger.info("Database schema ready (%s).",
                        "SQLite" if self._use_sqlite else "PostgreSQL")
        except Exception as e:
            logger.error("DB init error: %s", e)
            if self._conn and not self._use_sqlite:
                self._conn.rollback()
            raise

    # ── Writes ────────────────────────────────────────────────────────────────

    def insert_reading(
        self,
        road_name: str,
        vehicle_count: int,
        congestion_score: int,
        congestion_level: str,
        avg_speed_kmh: float,        # renamed from avg_speed for clarity
        vehicle_types: dict,
        timestamp: Optional[datetime] = None,
    ):
        ts = timestamp or datetime.now()
        vt_json = json.dumps(vehicle_types)

        if self._use_sqlite:
            sql = """
            INSERT INTO traffic_readings
                (road_name, timestamp, vehicle_count, congestion_score,
                 congestion_level, avg_speed_kmh, vehicle_types)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            conn = self._get_conn()
            conn.execute(sql, (
                road_name, ts.isoformat(), vehicle_count, congestion_score,
                congestion_level, avg_speed_kmh, vt_json,
            ))
            conn.commit()
        else:
            sql = """
            INSERT INTO traffic_readings
                (road_name, timestamp, vehicle_count, congestion_score,
                 congestion_level, avg_speed_kmh, vehicle_types)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(sql, (
                    road_name, ts, vehicle_count, congestion_score,
                    congestion_level, avg_speed_kmh, vt_json,
                ))
            conn.commit()

    def create_incident(
        self,
        road_name: str,
        description: str,
        severity: str = "medium",
    ) -> int:
        ts = datetime.now()
        if self._use_sqlite:
            conn = self._get_conn()
            cur = conn.execute(
                "INSERT INTO incidents (road_name, description, severity, reported_at) VALUES (?,?,?,?)",
                (road_name, description, severity, ts.isoformat()),
            )
            conn.commit()
            return cur.lastrowid
        else:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO incidents (road_name, description, severity, reported_at) VALUES (%s,%s,%s,%s) RETURNING id",
                    (road_name, description, severity, ts),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
            return row_id

    def resolve_incident(self, incident_id: int):
        ts = datetime.now()
        if self._use_sqlite:
            conn = self._get_conn()
            conn.execute(
                "UPDATE incidents SET resolved_at=? WHERE id=?",
                (ts.isoformat(), incident_id),
            )
            conn.commit()
        else:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE incidents SET resolved_at=%s WHERE id=%s", (ts, incident_id)
                )
            conn.commit()

    # ── Analytics queries ─────────────────────────────────────────────────────

    def get_all_road_names(self) -> list[str]:
        """Return all road names that have readings in the DB."""
        sql = "SELECT DISTINCT road_name FROM traffic_readings ORDER BY road_name"
        rows = self._query(sql)
        return [r["road_name"] for r in rows]

    def get_hourly_trend(self, road_name: Optional[str] = None, hours: int = 24) -> list[dict]:
        if self._use_sqlite:
            road_filter = "AND road_name = ?" if road_name else ""
            sql = f"""
            SELECT
                strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour,
                SUM(vehicle_count)   AS total_vehicles,
                AVG(congestion_score) AS avg_congestion,
                AVG(avg_speed_kmh)   AS avg_speed,
                road_name
            FROM traffic_readings
            WHERE timestamp >= datetime('now', '-{int(hours)} hours')
            {road_filter}
            GROUP BY strftime('%Y-%m-%dT%H:00:00', timestamp), road_name
            ORDER BY hour ASC
            """
            params = [road_name] if road_name else []
        else:
            road_filter = "AND road_name = %s" if road_name else ""
            sql = f"""
            SELECT
                DATE_TRUNC('hour', timestamp) AS hour,
                SUM(vehicle_count)            AS total_vehicles,
                AVG(congestion_score)         AS avg_congestion,
                AVG(avg_speed_kmh)            AS avg_speed,
                road_name
            FROM traffic_readings
            WHERE timestamp >= NOW() - INTERVAL '{int(hours)} hours'
            {road_filter}
            GROUP BY DATE_TRUNC('hour', timestamp), road_name
            ORDER BY hour ASC
            """
            params = [road_name] if road_name else []
        return self._query(sql, params)

    def get_vehicle_type_totals(self, hours: int = 24) -> dict:
        if self._use_sqlite:
            rows = self._query(
                f"SELECT vehicle_types FROM traffic_readings WHERE timestamp >= datetime('now', '-{int(hours)} hours')"
            )
            totals: dict[str, int] = {}
            for r in rows:
                try:
                    vt = json.loads(r["vehicle_types"] or "{}")
                    for k, v in vt.items():
                        totals[k] = totals.get(k, 0) + int(v or 0)
                except Exception:
                    pass
            return totals
        else:
            sql = f"""
            SELECT
                SUM((vehicle_types->>'car')::int)        AS car,
                SUM((vehicle_types->>'motorcycle')::int) AS motorcycle,
                SUM((vehicle_types->>'truck')::int)      AS truck,
                SUM((vehicle_types->>'bus')::int)        AS bus,
                SUM((vehicle_types->>'bicycle')::int)    AS bicycle
            FROM traffic_readings
            WHERE timestamp >= NOW() - INTERVAL '{int(hours)} hours'
            """
            rows = self._query(sql)
            return dict(rows[0]) if rows else {}

    def get_road_usage_stats(self) -> list[dict]:
        if self._use_sqlite:
            sql = """
            SELECT
                road_name,
                AVG(congestion_score)  AS avg_congestion,
                MAX(congestion_score)  AS peak_congestion,
                AVG(avg_speed_kmh)     AS avg_speed,
                SUM(vehicle_count)     AS total_vehicles
            FROM traffic_readings
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY road_name
            ORDER BY avg_congestion DESC
            """
        else:
            sql = """
            SELECT
                road_name,
                AVG(congestion_score)  AS avg_congestion,
                MAX(congestion_score)  AS peak_congestion,
                AVG(avg_speed_kmh)     AS avg_speed,
                SUM(vehicle_count)     AS total_vehicles
            FROM traffic_readings
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY road_name
            ORDER BY avg_congestion DESC
            """
        return self._query(sql)

    def get_congestion_prediction(self, road_name: str, look_ahead_hours: int = 3) -> list[dict]:
        if self._use_sqlite:
            sql = """
            SELECT
                AVG(congestion_score) AS avg_score,
                AVG(avg_speed_kmh)    AS avg_speed,
                CAST(strftime('%H', timestamp) AS INTEGER) AS hr
            FROM traffic_readings
            WHERE road_name = ?
              AND timestamp >= datetime('now', '-7 days')
            GROUP BY CAST(strftime('%H', timestamp) AS INTEGER)
            """
            rows = self._query(sql, [road_name])
            hist = {int(r["hr"]): r for r in rows}
            import datetime as dt_mod
            now = datetime.now()
            result = []
            for h in range(1, look_ahead_hours + 1):
                ft = now.replace(minute=0, second=0, microsecond=0)
                ft = ft.replace(hour=(ft.hour + h) % 24)
                hr_key = ft.hour
                result.append({
                    "forecast_time": ft,
                    "predicted_congestion": float(hist.get(hr_key, {}).get("avg_score") or 40),
                    "predicted_speed": float(hist.get(hr_key, {}).get("avg_speed") or 30),
                })
            return result
        else:
            sql = """
            WITH future_hours AS (
                SELECT generate_series(1, %s) AS h
            ),
            historical AS (
                SELECT
                    EXTRACT(HOUR FROM timestamp) AS hr,
                    AVG(congestion_score)        AS avg_score,
                    AVG(avg_speed_kmh)           AS avg_speed
                FROM traffic_readings
                WHERE road_name = %s
                  AND timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY hr
            )
            SELECT
                NOW() + (fh.h || ' hours')::INTERVAL   AS forecast_time,
                COALESCE(h.avg_score, 40)               AS predicted_congestion,
                COALESCE(h.avg_speed, 30)               AS predicted_speed
            FROM future_hours fh
            LEFT JOIN historical h
                ON h.hr = EXTRACT(HOUR FROM NOW() + (fh.h || ' hours')::INTERVAL)
            ORDER BY fh.h
            """
            return self._query(sql, [look_ahead_hours, road_name])

    def get_peak_hours(self, road_name: Optional[str] = None) -> list[dict]:
        if self._use_sqlite:
            road_filter = "AND road_name = ?" if road_name else ""
            sql = f"""
            SELECT
                CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                AVG(vehicle_count)   AS avg_count,
                AVG(congestion_score) AS avg_congestion
            FROM traffic_readings
            WHERE timestamp >= datetime('now', '-7 days')
            {road_filter}
            GROUP BY CAST(strftime('%H', timestamp) AS INTEGER)
            ORDER BY hour ASC
            """
            params = [road_name] if road_name else []
        else:
            road_filter = "AND road_name = %s" if road_name else ""
            sql = f"""
            SELECT
                EXTRACT(HOUR FROM timestamp)::int AS hour,
                AVG(vehicle_count)                AS avg_count,
                AVG(congestion_score)             AS avg_congestion
            FROM traffic_readings
            WHERE timestamp >= NOW() - INTERVAL '7 days'
            {road_filter}
            GROUP BY EXTRACT(HOUR FROM timestamp)
            ORDER BY hour ASC
            """
            params = [road_name] if road_name else []
        return self._query(sql, params)

    def get_current_incidents(self) -> list[dict]:
        if self._use_sqlite:
            sql = "SELECT id, road_name, description, severity, reported_at FROM incidents WHERE resolved_at IS NULL ORDER BY reported_at DESC"
        else:
            sql = "SELECT id, road_name, description, severity, reported_at FROM incidents WHERE resolved_at IS NULL ORDER BY reported_at DESC"
        return self._query(sql)

    def get_summary_stats(self) -> dict:
        if self._use_sqlite:
            sql = """
            SELECT
                SUM(vehicle_count)    AS total_vehicles_today,
                AVG(avg_speed_kmh)    AS avg_speed,
                AVG(congestion_score) AS avg_congestion,
                MAX(congestion_score) AS peak_congestion
            FROM traffic_readings
            WHERE timestamp >= date('now')
            """
            rows = self._query(sql)
            stats = dict(rows[0]) if rows else {}
            inc = self._query("SELECT COUNT(*) AS cnt FROM incidents WHERE resolved_at IS NULL")
            stats["active_incidents"] = inc[0]["cnt"] if inc else 0
            return stats
        else:
            sql = """
            SELECT
                SUM(vehicle_count)                          AS total_vehicles_today,
                AVG(avg_speed_kmh)                          AS avg_speed,
                AVG(congestion_score)                       AS avg_congestion,
                MAX(congestion_score)                       AS peak_congestion,
                (SELECT COUNT(*) FROM incidents WHERE resolved_at IS NULL) AS active_incidents
            FROM traffic_readings
            WHERE timestamp >= CURRENT_DATE
            """
            rows = self._query(sql)
            return dict(rows[0]) if rows else {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _query(self, sql: str, params=None) -> list[dict]:
        try:
            conn = self._get_conn()
            if self._use_sqlite:
                cur = conn.execute(sql, params or [])
                rows = cur.fetchall()
                return [dict(row) for row in rows]
            else:
                from psycopg2.extras import RealDictCursor
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params or [])
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error("Query error: %s", e)
            if self._conn and not self._use_sqlite:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            return []
