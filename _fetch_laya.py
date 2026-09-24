"""Resumable direct download of convaiinnovations/laya via hf-mirror.com.

Downloads every repo file into .models/laya/ next to this script. Each file
resumes with HTTP Range on any stall or drop, so no progress is ever lost.
"""
import json
import os
import time
import urllib.request

BASE = "https://hf-mirror.com/convaiinnovations/laya/resolve/main/"
API = "https://hf-mirror.com/api/models/convaiinnovations/laya/tree/main?recursive=true"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".models", "laya")
CHUNK = 1 << 20


def list_files() -> list:
    for attempt in range(20):
        try:
            with urllib.request.urlopen(API, timeout=30) as r:
                items = json.loads(r.read().decode())
            return [it for it in items if it.get("type") == "file"]
        except Exception as e:  # noqa: BLE001 - mirror connection is flaky
            wait = min(60, 5 * (attempt + 1))
            print(f"list_files retry {attempt + 1} ({type(e).__name__}), "
                  f"wait {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("could not list repo files")


def fetch(path: str, total: int) -> None:
    dest = os.path.join(DEST, path.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    fails = 0
    while done < total:
        base = BASE if fails // 5 % 2 == 0 else "https://huggingface.co/convaiinnovations/laya/resolve/main/"
        req = urllib.request.Request(
            base + path,
            headers={"Range": f"bytes={done}-", "User-Agent": "jev-pendulum"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                mode = "ab" if done else "wb"
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if done % (64 << 20) < CHUNK:
                            print(f"  {path}: {done / 1e6:.0f}/{total / 1e6:.0f} MB",
                                  flush=True)
            # stream ended; loop resumes if short
            fails = 0
        except Exception as e:  # noqa: BLE001 - reconnect and resume
            fails += 1
            wait = min(60, 5 * fails)
            print(f"  {path}: reconnect at {done / 1e6:.0f} MB "
                  f"({type(e).__name__}), wait {wait}s", flush=True)
            time.sleep(wait)
        done = os.path.getsize(dest) if os.path.exists(dest) else 0
    print(f"  {path}: complete ({total / 1e6:.1f} MB)", flush=True)


def main() -> None:
    files = list_files()
    print(f"{len(files)} files", flush=True)
    for i, it in enumerate(files, 1):
        if it["path"] == "multilingual/model.safetensors":
            continue  # second 644 MB checkpoint; we only use the main one
        size = int(it.get("size", 0))
        print(f"[{i}/{len(files)}] {it['path']} ({size / 1e6:.1f} MB)", flush=True)
        fetch(it["path"], size)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
