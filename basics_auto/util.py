"""Small, local JSON report writer for selected BASICS review utilities."""
import json
import os
from pathlib import Path
from uuid import uuid4

def atomic_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
