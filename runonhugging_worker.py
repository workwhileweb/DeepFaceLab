import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

_omp_raw = os.environ.get("OMP_NUM_THREADS", "")
if _omp_raw and not _omp_raw.isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

from autotrain import run_pipeline


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _read_status(job_dir: Path) -> dict:
    return json.loads(_status_path(job_dir).read_text(encoding="utf-8"))


def _write_status(job_dir: Path, payload: dict) -> None:
    _status_path(job_dir).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--max-hours", type=float, required=True)
    parser.add_argument("--plateau-hours", type=float, required=True)
    args = parser.parse_args()

    job_dir = Path(args.job_dir)
    st = _read_status(job_dir)

    try:
        env_overrides = {
            "AUTOTRAIN_WORKDIR": str(job_dir / "runs"),
            "AUTOTRAIN_MAX_HOURS": str(args.max_hours),
            "AUTOTRAIN_PLATEAU_HOURS": str(args.plateau_hours),
            "AUTOTRAIN_BACKUP_MIN": "10",
            "AUTOTRAIN_SAVE_MIN": "10",
            "AUTOTRAIN_ENABLE_XSEG": "1",
            "AUTOTRAIN_ENABLE_ENHANCE": "1",
        }
        result = run_pipeline([args.src, args.dst], preset=args.preset, env_overrides=env_overrides)
        run_dir = Path(result["run_dir"])
        summary_file = run_dir / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        st["status"] = "completed"
        st["updated_at"] = _now()
        st["message"] = "Completed"
        st["run_dir"] = str(run_dir)
        st["result_dfm"] = summary.get("dfm_output")
        _write_status(job_dir, st)
        return 0
    except Exception as e:
        st["status"] = "failed"
        st["updated_at"] = _now()
        st["message"] = f"{type(e).__name__}: {e}"
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        _write_status(job_dir, st)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
