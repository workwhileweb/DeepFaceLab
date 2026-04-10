import json
import os
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

# Hugging Face sets OMP_NUM_THREADS from CPU limits (e.g. "3500m"), while
# numexpr expects a plain integer. Normalize before importing DeepFaceLab stack.
_omp_raw = os.environ.get("OMP_NUM_THREADS", "")
if _omp_raw and not _omp_raw.isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

APP_ROOT = Path(__file__).resolve().parent
# Use explicit override first. For HF bucket-mounted /app, /tmp is often
# more stable for heavy write workloads during training.
_data_root_env = os.environ.get("DFL_DATA_ROOT", "").strip()
if _data_root_env:
    DATA_ROOT = Path(_data_root_env)
elif Path("/data").exists():
    DATA_ROOT = Path("/data")
else:
    DATA_ROOT = APP_ROOT
SERVICE_ROOT = DATA_ROOT / "hf_service"
JOBS_DIR = SERVICE_ROOT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DeepFaceLab Hugging AutoTrain API", version="1.0.0")


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    message: str = ""
    run_dir: Optional[str] = None
    result_dfm: Optional[str] = None


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _write_status(job_dir: Path, payload: dict) -> None:
    _status_path(job_dir).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_status(job_dir: Path) -> dict:
    p = _status_path(job_dir)
    if not p.exists():
        raise FileNotFoundError("status.json not found")
    return json.loads(p.read_text(encoding="utf-8"))


@app.on_event("startup")
def _recover_interrupted_jobs() -> None:
    # If app restarts, background threads are gone. Mark running jobs accordingly.
    for job_dir in JOBS_DIR.glob("*"):
        if not job_dir.is_dir():
            continue
        status_file = _status_path(job_dir)
        if not status_file.exists():
            continue
        try:
            st = _read_status(job_dir)
        except Exception:
            continue
        if st.get("status") in {"queued", "running"}:
            st["status"] = "failed"
            st["updated_at"] = _now()
            st["message"] = "Service restarted before job completion. Please resubmit."
            _write_status(job_dir, st)


def _run_job(job_id: str, src_path: Path, dst_path: Path, preset: str, max_hours: float, plateau_hours: float) -> None:
    job_dir = JOBS_DIR / job_id
    cmd = [
        "python3",
        "runonhugging_worker.py",
        "--job-dir",
        str(job_dir),
        "--src",
        str(src_path),
        "--dst",
        str(dst_path),
        "--preset",
        preset,
        "--max-hours",
        str(max_hours),
        "--plateau-hours",
        str(plateau_hours),
    ]
    proc = subprocess.Popen(cmd, cwd=str(APP_ROOT))
    st = _read_status(job_dir)
    st["status"] = "running"
    st["updated_at"] = _now()
    st["message"] = "Pipeline started"
    st["worker_pid"] = proc.pid
    _write_status(job_dir, st)


@app.get("/healthz")
def healthz():
    return {"ok": True, "data_root": str(DATA_ROOT), "jobs_dir": str(JOBS_DIR)}


@app.post("/jobs", response_model=JobStatus)
async def create_job(
    src_video: UploadFile = File(...),
    dst_video: UploadFile = File(...),
    preset: str = Form("balanced"),
    max_hours: float = Form(1.0),
    plateau_hours: float = Form(1.0),
):
    if preset not in {"fast", "balanced", "quality"}:
        raise HTTPException(status_code=400, detail="preset must be one of: fast, balanced, quality")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    src_path = input_dir / src_video.filename
    dst_path = input_dir / dst_video.filename
    src_path.write_bytes(await src_video.read())
    dst_path.write_bytes(await dst_video.read())

    status = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "message": "Queued",
        "run_dir": None,
        "result_dfm": None,
    }
    _write_status(job_dir, status)

    t = threading.Thread(
        target=_run_job,
        args=(job_id, src_path, dst_path, preset, max_hours, plateau_hours),
        daemon=True,
    )
    t.start()
    return JobStatus(**status)


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    st = _read_status(job_dir)
    pid = st.get("worker_pid")
    if st.get("status") == "running" and isinstance(pid, int):
        try:
            os.kill(pid, 0)
        except OSError:
            st["status"] = "failed"
            st["updated_at"] = _now()
            st["message"] = "Worker process exited unexpectedly."
            _write_status(job_dir, st)
    return JobStatus(**st)


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail_lines: int = 200):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")

    st = _read_status(job_dir)
    run_dir = st.get("run_dir")
    log_file = Path(run_dir) / "autotrain.log" if run_dir else None
    if (log_file is None) or (not log_file.exists()):
        runs_dir = job_dir / "runs"
        if runs_dir.exists():
            candidates = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                candidate_log = candidates[0] / "autotrain.log"
                if candidate_log.exists():
                    log_file = candidate_log
    if log_file is None or not log_file.exists():
        return PlainTextResponse("Run has not started yet.")

    lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return PlainTextResponse("\n".join(lines[-max(1, tail_lines) :]))


@app.get("/jobs/{job_id}/download")
def download_model(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    st = _read_status(job_dir)
    dfm = st.get("result_dfm")
    if not dfm:
        raise HTTPException(status_code=404, detail="result dfm not available")
    p = Path(dfm)
    if not p.exists():
        raise HTTPException(status_code=404, detail="dfm file missing on disk")
    return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)
