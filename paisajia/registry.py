"""Paso 5 del flujo: registro estructurado en SQLite y JSON por ronda.

Se guardan decisiones, motivos, tiempos, semillas y parámetros. Por privacidad
el registro guarda el hash SHA-256 de la foto, nunca la foto.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH, PRESERVE_SUFFIX, SESSIONS_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    image_hash TEXT NOT NULL,
    scene TEXT,
    generator TEXT,
    llm TEXT,
    simulated INTEGER DEFAULT 0,
    closed_at REAL
);
CREATE TABLE IF NOT EXISTS rounds (
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    started_at REAL NOT NULL,
    gen_seconds REAL,
    lpips_mean REAL,
    lpips_backend TEXT,
    package_comment TEXT,
    closed_at REAL,
    PRIMARY KEY (session_id, round)
);
CREATE TABLE IF NOT EXISTS variations (
    var_id TEXT NOT NULL,           -- p. ej. R2-3 (ronda 2, posición 3)
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    slot INTEGER NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,  -- aumenta al regenerar
    active INTEGER NOT NULL DEFAULT 1,
    axis TEXT NOT NULL,
    variant TEXT NOT NULL,
    prompt TEXT NOT NULL,
    seed INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    dino REAL,
    dino_backend TEXT,
    clip_adherence REAL,
    warning TEXT,
    plan_reason TEXT,
    params TEXT,
    gen_seconds REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, var_id, version)
);
CREATE TABLE IF NOT EXISTS decisions (
    session_id TEXT NOT NULL,
    var_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('aceptada','rechazada')),
    reason TEXT,
    via TEXT NOT NULL,              -- individual | paquete
    shown_at REAL,
    decided_at REAL NOT NULL,
    PRIMARY KEY (session_id, var_id)
);
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    llm TEXT,
    attempts INTEGER,
    first_try_valid INTEGER,
    used_template INTEGER,
    errors TEXT,
    output TEXT,
    seconds REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS guardrail_events (
    session_id TEXT,
    kind TEXT NOT NULL,             -- entrada | instruccion | salida
    category TEXT,
    detail TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS surveys (
    survey_id TEXT PRIMARY KEY,
    session_id TEXT,
    sus_answers TEXT NOT NULL,
    sus_score REAL NOT NULL,
    clarity INTEGER,
    fidelity INTEGER,
    comment TEXT,
    created_at REAL NOT NULL
);
"""


class Registry:
    def __init__(self, db_path: Path | str = DB_PATH, sessions_dir: Path | str = SESSIONS_DIR):
        self.db_path = Path(db_path)
        self.sessions_dir = Path(sessions_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    # -- sesiones ---------------------------------------------------------
    def create_session(self, image_hash: str, scene: str | None, generator: str, llm: str, simulated: bool = False) -> str:
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self._conn() as c:
            c.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,NULL)",
                (sid, time.time(), image_hash, scene, generator, llm, int(simulated)),
            )
        self.session_dir(sid).mkdir(parents=True, exist_ok=True)
        return sid

    def session_dir(self, sid: str) -> Path:
        return self.sessions_dir / sid

    def close_session(self, sid: str, delete_files: bool = True) -> None:
        """Cierra la sesión. Por privacidad, borra las imágenes generadas del disco."""
        with self._conn() as c:
            c.execute("UPDATE sessions SET closed_at=? WHERE session_id=?", (time.time(), sid))
        if delete_files:
            d = self.session_dir(sid)
            for p in d.glob("*.png"):
                p.unlink(missing_ok=True)

    def purge_session(self, sid: str) -> None:
        """Borrado completo (derecho de supresión, Ley 29733)."""
        with self._conn() as c:
            for t in ("sessions", "rounds", "variations", "decisions", "agent_runs", "guardrail_events", "surveys"):
                c.execute(f"DELETE FROM {t} WHERE session_id=?", (sid,))
        shutil.rmtree(self.session_dir(sid), ignore_errors=True)

    # -- rondas y variaciones --------------------------------------------
    def start_round(self, sid: str, rnd: int) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO rounds (session_id, round, started_at) VALUES (?,?,?)", (sid, rnd, time.time()))

    def finish_generation(self, sid: str, rnd: int, seconds: float, lpips_mean: float, lpips_backend: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE rounds SET gen_seconds=?, lpips_mean=?, lpips_backend=? WHERE session_id=? AND round=?",
                (seconds, lpips_mean, lpips_backend, sid, rnd),
            )

    def add_variation(self, sid: str, v: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE variations SET active=0 WHERE session_id=? AND var_id=?", (sid, v["var_id"]))
            c.execute(
                """INSERT INTO variations (var_id, session_id, round, slot, version, active, axis, variant, prompt,
                   seed, attempts, dino, dino_backend, clip_adherence, warning, plan_reason, params, gen_seconds, created_at)
                   VALUES (?,?,?,?,?,1,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (v["var_id"], sid, v["round"], v["slot"], v.get("version", 1), v["axis"], v["variant"], v["prompt"],
                 v["seed"], v["attempts"], v.get("dino"), v.get("dino_backend"), v.get("clip_adherence"),
                 v.get("warning"), v.get("plan_reason"), json.dumps(v.get("params", {}), ensure_ascii=False),
                 v.get("gen_seconds"), time.time()),
            )

    def record_decision(self, sid: str, var_id: str, rnd: int, decision: str, reason: str | None,
                        via: str = "individual", shown_at: float | None = None) -> None:
        if decision not in ("aceptada", "rechazada"):
            raise ValueError(decision)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)",
                (sid, var_id, rnd, decision, (reason or "").strip() or None, via, shown_at, time.time()),
            )

    def clear_decision(self, sid: str, var_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM decisions WHERE session_id=? AND var_id=?", (sid, var_id))

    def set_package_comment(self, sid: str, rnd: int, comment: str | None) -> None:
        with self._conn() as c:
            c.execute("UPDATE rounds SET package_comment=? WHERE session_id=? AND round=?", ((comment or "").strip() or None, sid, rnd))

    def log_guardrail(self, sid: str | None, kind: str, category: str | None, detail: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO guardrail_events VALUES (?,?,?,?,?)", (sid, kind, category, detail, time.time()))

    def log_agent_run(self, sid: str, rnd: int, llm: str, attempts: int, first_try_valid: bool,
                      used_template: bool, errors: list[str], output: dict, seconds: float) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO agent_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (uuid.uuid4().hex, sid, rnd, llm, attempts, int(first_try_valid), int(used_template),
                 json.dumps(errors, ensure_ascii=False), json.dumps(output, ensure_ascii=False), seconds, time.time()),
            )
        with self._conn() as c:
            c.execute("UPDATE rounds SET closed_at=? WHERE session_id=? AND round=?", (time.time(), sid, rnd))

    def save_survey(self, sid: str | None, answers: list[int], sus: float, clarity: int, fidelity: int, comment: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO surveys VALUES (?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, sid, json.dumps(answers), sus, clarity, fidelity, comment, time.time()))

    # -- lectura -----------------------------------------------------------
    def decision_log(self, sid: str, up_to_round: int | None = None) -> list[dict]:
        """Registro que recibe el agente: IDs, ejes, variantes, decisiones y motivos (nunca la foto)."""
        sql = """SELECT v.var_id AS id, v.round AS ronda, v.axis AS eje, v.variant AS variante,
                        CASE WHEN v.axis='personalizada' THEN v.prompt END AS instruccion,
                        d.decision AS decision, d.reason AS motivo, d.via AS via
                 FROM variations v LEFT JOIN decisions d ON d.session_id=v.session_id AND d.var_id=v.var_id
                 WHERE v.session_id=? AND v.active=1"""
        params: tuple = (sid,)
        if up_to_round is not None:
            sql += " AND v.round<=?"
            params = (sid, up_to_round)
        sql += " ORDER BY v.round, v.slot"
        rows = self.query(sql, params)
        for r in rows:
            if r["instruccion"]:
                r["instruccion"] = r["instruccion"].replace(PRESERVE_SUFFIX, "").strip()
        return rows

    def package_comments(self, sid: str) -> dict[int, str]:
        rows = self.query("SELECT round, package_comment FROM rounds WHERE session_id=? AND package_comment IS NOT NULL", (sid,))
        return {r["round"]: r["package_comment"] for r in rows}

    def variant_stats(self, sid: str) -> dict[tuple[str, str], tuple[int, int]]:
        """(aceptadas, rechazadas) por (eje, variante) en la sesión: alimenta el bandido."""
        rows = self.query(
            """SELECT v.axis, v.variant, d.decision, COUNT(*) AS n FROM variations v
               JOIN decisions d ON d.session_id=v.session_id AND d.var_id=v.var_id
               WHERE v.session_id=? AND v.active=1 GROUP BY v.axis, v.variant, d.decision""",
            (sid,),
        )
        stats: dict[tuple[str, str], list[int]] = {}
        for r in rows:
            s = stats.setdefault((r["axis"], r["variant"]), [0, 0])
            s[0 if r["decision"] == "aceptada" else 1] += r["n"]
        return {k: (a, b) for k, (a, b) in stats.items()}

    def export_round_json(self, sid: str, rnd: int, extra: dict | None = None) -> Path:
        variations = self.query("SELECT * FROM variations WHERE session_id=? AND round=? ORDER BY slot, version", (sid, rnd))
        for v in variations:
            v["params"] = json.loads(v["params"] or "{}")
        data = {
            "session": self.query("SELECT * FROM sessions WHERE session_id=?", (sid,))[0],
            "round": self.query("SELECT * FROM rounds WHERE session_id=? AND round=?", (sid, rnd))[0],
            "variations": variations,
            "decisions": self.query("SELECT * FROM decisions WHERE session_id=? AND round=? ORDER BY var_id", (sid, rnd)),
            "agent_runs": self.query("SELECT * FROM agent_runs WHERE session_id=? AND round=?", (sid, rnd)),
            **(extra or {}),
        }
        path = self.session_dir(sid) / f"ronda_{rnd}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
