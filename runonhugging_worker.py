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
    parser.add_argument("--job-id", required=True)
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
            "NN_KEEP_CUDA_VISIBLE_DEVICES": os.environ.get("NN_KEEP_CUDA_VISIBLE_DEVICES", "1"),
            "AUTOTRAIN_WORKDIR": str(job_dir / "runs"),
            "AUTOTRAIN_MAX_HOURS": str(args.max_hours),
            "AUTOTRAIN_PLATEAU_HOURS": str(args.plateau_hours),
            "AUTOTRAIN_BACKUP_MIN": "10",
            "AUTOTRAIN_SAVE_MIN": "10",
            "AUTOTRAIN_MIN_ITERS": os.environ.get("AUTOTRAIN_MIN_ITERS", "2000"),
            "AUTOTRAIN_MIN_SRC_FACES": os.environ.get("AUTOTRAIN_MIN_SRC_FACES", "300"),
            "AUTOTRAIN_MIN_DST_FACES": os.environ.get("AUTOTRAIN_MIN_DST_FACES", "300"),
            "AUTOTRAIN_ENABLE_XSEG": "1",
            "AUTOTRAIN_ENABLE_ENHANCE": "1",
            # Speed-focused defaults for stronger GPU hardware.
            "AUTOTRAIN_PREPROCESS_PARALLEL": os.environ.get("AUTOTRAIN_PREPROCESS_PARALLEL", "1"),
            "AUTOTRAIN_PARALLEL_ALIGN": os.environ.get("AUTOTRAIN_PARALLEL_ALIGN", "0"),
            "AUTOTRAIN_PARALLEL_ENHANCE": os.environ.get("AUTOTRAIN_PARALLEL_ENHANCE", "0"),
            "AUTOTRAIN_BATCH_SIZE": os.environ.get("AUTOTRAIN_BATCH_SIZE", "12"),
        }
        result = run_pipeline([args.src, args.dst], preset=args.preset, env_overrides=env_overrides)
        run_dir = Path(result["run_dir"])
        summary_file = run_dir / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        quality_report = run_dir / "reports" / "quality_report.json"
        quality_data = {}
        if quality_report.exists():
            quality_data = json.loads(quality_report.read_text(encoding="utf-8"))
        st["status"] = "completed"
        st["updated_at"] = _now()
        st["message"] = "Completed"
        st["run_dir"] = str(run_dir)
        st["result_dfm"] = summary.get("dfm_output")
        st["progress"] = {
            "phase": "completed",
            "run_id": run_dir.name,
            "quality": quality_data,
        }
        _write_status(job_dir, st)
        timeline = job_dir / "timeline.md"
        timeline.write_text(
            timeline.read_text(encoding="utf-8") + f"- `{_now()}` Job `{args.job_id}` completed successfully\n"
            if timeline.exists()
            else f"- `{_now()}` Job `{args.job_id}` completed successfully\n",
            encoding="utf-8",
        )
        return 0
    except Exception as e:
        if st.get("status") == "cancelling":
            st["status"] = "cancelled"
            st["message"] = "Cancelled"
        else:
            st["status"] = "failed"
            st["message"] = f"{type(e).__name__}: {e}"
        st["updated_at"] = _now()
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        st["progress"] = {
            "phase": st["status"],
            "error_log": str(job_dir / "error.log"),
        }
        _write_status(job_dir, st)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
