from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import tempfile

def save_json(payload: dict, output_dir: str | Path = "output", name: str | None = None) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = name or f"vn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    final_path = out_dir / filename
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=out_dir, suffix=".tmp") as tf:
        json.dump(payload, tf, ensure_ascii=False, indent=2)
        tmp_path = Path(tf.name)
    tmp_path.replace(final_path)
    return final_path