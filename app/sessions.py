"""In-memory conversation sessions: the order so far, per phone call.

Good enough for one process. In production this would live in Redis (or similar)
so that several server instances share sessions and they survive restarts.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    id: str
    order: dict[str, int] = field(default_factory=dict)  # menu id -> quantity
    turns: int = 0
    last_seen: float = field(default_factory=time.monotonic)


class SessionStore:
    def __init__(self, ttl_s: float = 30 * 60, max_sessions: int = 1000):
        self._ttl_s = ttl_s
        self._max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        with self._lock:
            self._evict()
            session = Session(id=uuid.uuid4().hex)
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._evict()
            session = self._sessions.get(session_id)
            if session:
                session.last_seen = time.monotonic()
            return session

    def save(self, session: Session) -> None:
        with self._lock:
            session.last_seen = time.monotonic()
            self._sessions[session.id] = session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict(self) -> None:
        now = time.monotonic()
        for sid in [sid for sid, s in self._sessions.items() if now - s.last_seen > self._ttl_s]:
            del self._sessions[sid]
        while len(self._sessions) >= self._max_sessions:  # drop the least recently used
            oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
            del self._sessions[oldest.id]
