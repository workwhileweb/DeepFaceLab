import contextlib
import json
import os
import signal
import subprocess
import threading
import uuid
import zipfile
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
WORKER_SCRIPT = APP_ROOT / "runonhugging_worker.py"
JOB_LOCK = threading.Lock()
QUEUE_CHECK_INTERVAL_SEC = max(3, int(os.environ.get("HF_QUEUE_SCAN_SECONDS", "5")))


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    message: str = ""
    run_dir: Optional[str] = None
    result_dfm: Optional[str] = None
    worker_pid: Optional[int] = None
    progress: Optional[dict] = None


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _status_path(job_dir: Path) -> Path:
    return job_dir / "status.json"


def _write_status(job_dir: Path, payload: dict) -> None:
    payload.setdefault("updated_at", _now())
    _status_path(job_dir).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_status(job_dir: Path) -> dict:
    p = _status_path(job_dir)
    if not p.exists():
        raise FileNotFoundError("status.json not found")
    return json.loads(p.read_text(encoding="utf-8"))


def _append_timeline(job_dir: Path, message: str) -> None:
    timeline = job_dir / "timeline.md"
    timeline.parent.mkdir(parents=True, exist_ok=True)
    with timeline.open("a", encoding="utf-8") as f:
        f.write(f"- `{_now()}` {message}\n")


def _job_dirs() -> list[Path]:
    return sorted([p for p in JOBS_DIR.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _build_worker_cmd(job_id: str, st: dict, job_dir: Path) -> list[str]:
    src_path = st.get("src_path")
    dst_path = st.get("dst_path")
    if not src_path or not dst_path:
        raise RuntimeError("Missing src_path or dst_path in status.json")
    return [
        "python3",
        str(WORKER_SCRIPT),
        "--job-id",
        job_id,
        "--job-dir",
        str(job_dir),
        "--src",
        str(src_path),
        "--dst",
        str(dst_path),
        "--preset",
        str(st.get("preset", "balanced")),
        "--max-hours",
        str(st.get("max_hours", 3.0)),
        "--plateau-hours",
        str(st.get("plateau_hours", 2.0)),
    ]


def _launch_job_locked(job_dir: Path, st: dict) -> None:
    if st.get("status") != "queued":
        return
    running = []
    for jd in _job_dirs():
        try:
            cur = _read_status(jd)
        except Exception:
            continue
        pid = cur.get("worker_pid")
        if cur.get("status") == "running" and isinstance(pid, int) and _is_pid_running(pid):
            running.append(cur.get("job_id", jd.name))
    if running:
        return

    cmd = _build_worker_cmd(job_dir.name, st, job_dir)
    proc = subprocess.Popen(cmd, cwd=str(APP_ROOT))
    st["status"] = "running"
    st["updated_at"] = _now()
    st["message"] = "Pipeline started"
    st["worker_pid"] = proc.pid
    st["progress"] = {"phase": "running", "heartbeat_at": _now()}
    _write_status(job_dir, st)
    _append_timeline(job_dir, f"Worker started with pid `{proc.pid}`")


def _queue_loop() -> None:
    while True:
        try:
            with JOB_LOCK:
                for job_dir in _job_dirs():
                    try:
                        st = _read_status(job_dir)
                    except Exception:
                        continue
                    pid = st.get("worker_pid")
                    if st.get("status") == "running" and isinstance(pid, int):
                        if _is_pid_running(pid):
                            progress = st.get("progress") or {}
                            progress["phase"] = "running"
                            progress["heartbeat_at"] = _now()
                            st["progress"] = progress
                            st["updated_at"] = _now()
                            _write_status(job_dir, st)
                        else:
                            st["status"] = "failed"
                            st["message"] = "Worker process exited unexpectedly."
                            st["updated_at"] = _now()
                            st["worker_pid"] = None
                            _write_status(job_dir, st)
                            _append_timeline(job_dir, "Worker exited unexpectedly during queue heartbeat")
                    if st.get("status") == "queued":
                        _launch_job_locked(job_dir, st)
                        break
        except Exception:
            pass
        threading.Event().wait(QUEUE_CHECK_INTERVAL_SEC)


def _preflight() -> dict:
    gpu_ok = False
    gpu_info = "nvidia-smi unavailable"
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            gpu_ok = True
            gpu_info = proc.stdout.strip()
        else:
            gpu_info = (proc.stderr or proc.stdout or "").strip() or gpu_info
    except Exception as e:
        gpu_info = f"preflight error: {type(e).__name__}: {e}"
    return {
        "gpu_ok": gpu_ok,
        "gpu_info": gpu_info,
        "data_root_exists": DATA_ROOT.exists(),
        "jobs_dir_exists": JOBS_DIR.exists(),
        "worker_script_exists": WORKER_SCRIPT.exists(),
    }


def _all_jobs_terminal() -> bool:
    any_job = False
    for job_dir in _job_dirs():
        any_job = True
        try:
            st = _read_status(job_dir)
        except Exception:
            return False
        if st.get("status") in {"queued", "running", "cancelling"}:
            return False
    return any_job


def _sleep_guard_loop() -> None:
    if os.environ.get("HF_EXIT_WHEN_DONE", "1") != "1":
        return
    idle_checks = 0
    while True:
        threading.Event().wait(QUEUE_CHECK_INTERVAL_SEC)
        if _all_jobs_terminal():
            idle_checks += 1
            if idle_checks >= 3:
                os.kill(os.getpid(), signal.SIGTERM)
                return
        else:
            idle_checks = 0


@app.on_event("startup")
def _recover_interrupted_jobs() -> None:
    for job_dir in _job_dirs():
        try:
            st = _read_status(job_dir)
        except Exception:
            continue
        if st.get("status") == "running":
            st["updated_at"] = _now()
            st["status"] = "queued"
            st["message"] = "Recovered after service restart; re-queued."
            st.pop("worker_pid", None)
            _write_status(job_dir, st)
            _append_timeline(job_dir, "Service restarted, job re-queued automatically")

    threading.Thread(target=_queue_loop, daemon=True).start()
    threading.Thread(target=_sleep_guard_loop, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"ok": True, "data_root": str(DATA_ROOT), "jobs_dir": str(JOBS_DIR), "preflight": _preflight()}


@app.post("/jobs", response_model=JobStatus)
async def create_job(
    src_video: UploadFile = File(...),
    dst_video: UploadFile = File(...),
    preset: str = Form("balanced"),
    max_hours: float = Form(6.0),
    plateau_hours: float = Form(2.5),
):
    if preset not in {"fast", "balanced", "quality"}:
        raise HTTPException(status_code=400, detail="preset must be one of: fast, balanced, quality")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "inputs"
    runs_dir = job_dir / "runs"
    logs_dir = job_dir / "logs"
    reports_dir = job_dir / "reports"
    for p in (input_dir, runs_dir, logs_dir, reports_dir):
        p.mkdir(parents=True, exist_ok=True)

    src_name = Path(src_video.filename or "src.mp4").name
    dst_name = Path(dst_video.filename or "dst.mp4").name
    src_path = input_dir / src_name
    dst_path = input_dir / dst_name
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
        "preset": preset,
        "max_hours": max_hours,
        "plateau_hours": plateau_hours,
        "src_path": str(src_path),
        "dst_path": str(dst_path),
        "worker_pid": None,
        "progress": {
            "phase": "queued",
            "min_iterations": int(os.environ.get("AUTOTRAIN_MIN_ITERS", "2000")),
        },
    }
    _write_status(job_dir, status)
    _append_timeline(job_dir, f"Job created with preset `{preset}`")
    return JobStatus(**status)


@app.get("/jobs")
def list_jobs(limit: int = 50):
    items = []
    for job_dir in reversed(_job_dirs()):
        try:
            st = _read_status(job_dir)
            items.append(st)
        except Exception:
            continue
        if len(items) >= max(1, limit):
            break
    return items


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    st = _read_status(job_dir)
    pid = st.get("worker_pid")
    if st.get("status") == "running" and isinstance(pid, int):
        if not _is_pid_running(pid):
            st["status"] = "failed"
            st["updated_at"] = _now()
            st["message"] = "Worker process exited unexpectedly."
            _write_status(job_dir, st)
            _append_timeline(job_dir, "Worker process exited unexpectedly")
    return JobStatus(**st)


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail_lines: int = 200, offset: int = 0):
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
    start = max(0, offset)
    if tail_lines > 0:
        lines = lines[-max(1, tail_lines) :]
    if start > 0:
        lines = lines[start:]
    return PlainTextResponse("\n".join(lines))


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    with JOB_LOCK:
        st = _read_status(job_dir)
        pid = st.get("worker_pid")
        if st.get("status") not in {"queued", "running"}:
            raise HTTPException(status_code=409, detail=f"job is {st.get('status')}, cannot cancel")
        if isinstance(pid, int) and _is_pid_running(pid):
            os.kill(pid, signal.SIGTERM)
            st["message"] = "Cancellation requested"
            st["status"] = "cancelling"
        else:
            st["message"] = "Cancelled before worker start"
            st["status"] = "cancelled"
        st["updated_at"] = _now()
        _write_status(job_dir, st)
    _append_timeline(job_dir, "Cancellation requested")
    return st


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    st = _read_status(job_dir)
    if st.get("status") in {"queued", "running", "cancelling"}:
        raise HTTPException(status_code=409, detail="cannot delete active job")
    for path in sorted(job_dir.glob("**/*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file():
            path.unlink(missing_ok=True)
    for path in sorted(job_dir.glob("**/*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            with contextlib.suppress(Exception):
                path.rmdir()
    with contextlib.suppress(Exception):
        job_dir.rmdir()
    return {"ok": True, "job_id": job_id}


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


@app.get("/jobs/{job_id}/bundle")
def download_bundle(job_id: str):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    st = _read_status(job_dir)
    run_dir = st.get("run_dir")
    if not run_dir:
        raise HTTPException(status_code=404, detail="run dir not available")
    run_path = Path(run_dir)
    if not run_path.exists():
        raise HTTPException(status_code=404, detail="run dir missing on disk")
    bundle_dir = job_dir / "artifacts"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / f"{job_id}_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in (
            "summary.json",
            "summary.md",
            "metrics.json",
            "timeline.md",
            "reports/quality_report.json",
            "reports/quality_report.md",
        ):
            p = run_path / rel
            if p.exists():
                zf.write(p, arcname=rel)
        if st.get("result_dfm") and Path(st["result_dfm"]).exists():
            zf.write(st["result_dfm"], arcname=Path(st["result_dfm"]).name)
    return FileResponse(str(bundle_path), media_type="application/zip", filename=bundle_path.name)
