"""
PostgreSQL database manager using psycopg2.
Handles schema creation, inserts, and all analytics queries.
"""

import json
from datetime import datetime, timedelta
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
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

# No sample data seeding — all data comes from live camera sessions only.


class DatabaseManager:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            self._conn.autocommit = False
        return self._conn

    def initialize(self):
        """Create schema only — no sample data inserted."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
                conn.commit()
            logger.info("Database schema ready.")
        except Exception as e:
            logger.error(f"DB init error: {e}")
            if self._conn:
                self._conn.rollback()

    def insert_reading(
        self,
        road_name: str,
        vehicle_count: int,
        congestion_score: int,
        congestion_level: str,
        avg_speed: float,
        vehicle_types: dict,
        timestamp: Optional[datetime] = None,
    ):
        sql = """
        INSERT INTO traffic_readings
            (road_name, timestamp, vehicle_count, congestion_score,
             congestion_level, avg_speed_kmh, vehicle_types)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        ts = timestamp or datetime.now()
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, (
                road_name, ts, vehicle_count, congestion_score,
                congestion_level, avg_speed, json.dumps(vehicle_types)
            ))
        conn.commit()

    # ------------------------------------------------------------------
    # Analytics queries
    # ------------------------------------------------------------------
    def get_hourly_trend(self, road_name: Optional[str] = None, hours: int = 24) -> list[dict]:
        sql = """
        SELECT
            DATE_TRUNC('hour', timestamp) AS hour,
            SUM(vehicle_count)            AS total_vehicles,
            AVG(congestion_score)         AS avg_congestion,
            AVG(avg_speed_kmh)            AS avg_speed,
            road_name
        FROM traffic_readings
        WHERE timestamp >= NOW() - INTERVAL '%s hours'
        {road_filter}
        GROUP BY DATE_TRUNC('hour', timestamp), road_name
        ORDER BY hour ASC
        """
        road_filter = "AND road_name = %s" if road_name else ""
        params = [hours, road_name] if road_name else [hours]
        sql = sql.format(road_filter=road_filter)
        return self._query(sql, params)

    def get_vehicle_type_totals(self, hours: int = 24) -> dict:
        sql = """
        SELECT
            SUM((vehicle_types->>'car')::int)        AS car,
            SUM((vehicle_types->>'motorcycle')::int) AS motorcycle,
            SUM((vehicle_types->>'truck')::int)      AS truck,
            SUM((vehicle_types->>'bus')::int)        AS bus,
            SUM((vehicle_types->>'bicycle')::int)    AS bicycle
        FROM traffic_readings
        WHERE timestamp >= NOW() - INTERVAL '%s hours'
        """
        rows = self._query(sql, [hours])
        return dict(rows[0]) if rows else {}

    def get_road_usage_stats(self) -> list[dict]:
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
        """
        Simple prediction: use same-hour rolling average from the past 7 days
        as the forecast for the next N hours.
        """
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
        sql = """
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
        road_filter = "AND road_name = %s" if road_name else ""
        params = [road_name] if road_name else []
        sql = sql.format(road_filter=road_filter)
        return self._query(sql, params)

    def get_current_incidents(self) -> list[dict]:
        sql = """
        SELECT id, road_name, description, severity, reported_at
        FROM incidents
        WHERE resolved_at IS NULL
        ORDER BY reported_at DESC
        """
        return self._query(sql)

    def get_summary_stats(self) -> dict:
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

    # ------------------------------------------------------------------
    def _query(self, sql: str, params=None) -> list[dict]:
        try:
            conn = self._get_conn()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params or [])
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Query error: {e}")
            if self._conn:
                self._conn.rollback()
            return []

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
