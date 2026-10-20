from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3

from .models import Config, State, decimal, encode
from .analytics import remember_events


@contextmanager
def writer_lock(path: Path):
    """OS-held lock releases on process death, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(str(path) + ".lock", "a+b")
    if handle.seek(0, 2) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise ValueError("Эта база уже используется другим исполнителем") from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Store:
    def __init__(self, path: Path, *, readonly: bool = False):
        if readonly:
            self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, config TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS steps (
                session_id TEXT NOT NULL, step_id TEXT NOT NULL,
                PRIMARY KEY(session_id, step_id));
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, kind TEXT NOT NULL,
                occurred_at TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS analyses (
                session_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(session_id,kind));
        """)

    def close(self):
        self.db.close()

    def create(self, config: Config, state: State):
        with self.db:
            self.db.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                            (state.session_id, encode(config.to_dict()), encode(state.to_dict())))
            self.db.execute("INSERT INTO events(session_id,kind,occurred_at,payload) VALUES(?,?,?,?)",
                            (state.session_id, "SESSION_STARTED", state.started_at,
                             encode(config.to_dict())))

    def load(self, session_id: str) -> tuple[Config, State]:
        row = self.db.execute("SELECT config,state FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("Сессия не найдена")
        return Config.from_dict(json.loads(row[0])), State(**json.loads(row[1]))

    def seen(self, session_id: str, step_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM steps WHERE session_id=? AND step_id=?",
                               (session_id, step_id)).fetchone() is not None

    def commit_step(self, state: State, step_id: str, now: str, events: list[tuple[str, dict]]):
        # State, snapshot, verdict and fills are committed together in the paper model.
        with self.db:
            self.db.execute("INSERT INTO steps VALUES (?,?)", (state.session_id, step_id))
            remember_events(state, now, events)
            self.db.execute("UPDATE sessions SET state=? WHERE id=?",
                            (encode(state.to_dict()), state.session_id))
            self.db.executemany(
                "INSERT INTO events(session_id,kind,occurred_at,payload) VALUES(?,?,?,?)",
                [(state.session_id, kind, now, encode(payload)) for kind, payload in events])

    def events(self, session_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT seq,kind,occurred_at,payload FROM events WHERE session_id=? ORDER BY seq",
            (session_id,)).fetchall()
        return [dict(seq=r[0], kind=r[1], occurred_at=r[2], payload=json.loads(r[3])) for r in rows]

    def sessions(self) -> list[dict]:
        rows = self.db.execute("SELECT id,config,state FROM sessions ORDER BY rowid DESC").fetchall()
        return [dict(session_id=row[0], source=json.loads(row[1])["source"],
                     status=json.loads(row[2])["status"], started_at=json.loads(row[2])["started_at"])
                for row in rows]

    def delete_session(self, session_id: str):
        """Delete only a closed, flat archive, atomically with its dependent records.

        The caller must hold writer_lock so a separate executor cannot resume it.
        """
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            _, state = self.load(session_id)
            if (state.status != "CLOSED" or decimal(state.quantity) != 0
                    or (state.broker or {}).get("pending") is not None):
                raise ValueError("Удалить можно только завершённую сессию без позиции и незавершённой заявки")
            for table in ("analyses", "events", "steps", "sessions"):
                column = "id" if table == "sessions" else "session_id"
                self.db.execute(f"DELETE FROM {table} WHERE {column}=?", (session_id,))

    def save_analysis(self, session_id, kind, report):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO analyses VALUES (?,?,?)", (session_id,kind,encode(report)))

    def analysis(self, session_id, kind):
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='analyses'").fetchone():
            return None
        row = self.db.execute("SELECT payload FROM analyses WHERE session_id=? AND kind=?", (session_id,kind)).fetchone()
        return json.loads(row[0]) if row else None
