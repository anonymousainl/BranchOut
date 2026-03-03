from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import tempfile
from typing import Any, Dict


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as tf:
        json.dump(obj, tf, ensure_ascii=False, indent=2)
        tmp_path = Path(tf.name)
    tmp_path.replace(path)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class ArtifactStore:
    root: Path
    checkpoints_dir: Path = field(init=False)
    events_path: Path = field(init=False)
    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.checkpoints_dir = self.root / "checkpoints"
        self.events_path = self.root / "events.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in (name or ""))
        return safe[:120] if safe else "checkpoint"

    def checkpoint(self, name: str, payload: Any, *, also_latest: bool = True) -> Path:
        self._seq += 1
        safe = self._safe_name(name)
        p = self.checkpoints_dir / f"{self._seq:03d}_{safe}.json"
        _atomic_write_json(p, payload)
        if also_latest:
            _atomic_write_json(self.checkpoints_dir / "latest.json", payload)
        return p

    def save(self, rel_path: str, payload: Any) -> Path:
        p = self.root / rel_path
        _atomic_write_json(p, payload)
        return p

    def event(self, kind: str, data: Dict[str, Any]) -> None:
        _append_jsonl(
            self.events_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "kind": kind,
                "data": data,
            },
        )