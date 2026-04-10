import json
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from autotrain import run_pipeline


APP_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = APP_ROOT / "hf_service"
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


def _run_job(job_id: str, src_path: Path, dst_path: Path, preset: str, max_hours: float, plateau_hours: float) -> None:
    job_dir = JOBS_DIR / job_id
    try:
        st = _read_status(job_dir)
        st["status"] = "running"
        st["updated_at"] = _now()
        st["message"] = "Pipeline started"
        _write_status(job_dir, st)

        env_overrides = {
            "AUTOTRAIN_WORKDIR": str(job_dir / "runs"),
            "AUTOTRAIN_MAX_HOURS": str(max_hours),
            "AUTOTRAIN_PLATEAU_HOURS": str(plateau_hours),
            "AUTOTRAIN_BACKUP_MIN": "10",
            "AUTOTRAIN_SAVE_MIN": "10",
            "AUTOTRAIN_ENABLE_XSEG": "1",
            "AUTOTRAIN_ENABLE_ENHANCE": "1",
        }
        result = run_pipeline([str(src_path), str(dst_path)], preset=preset, env_overrides=env_overrides)

        run_dir = Path(result["run_dir"])
        summary_file = run_dir / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
        result_dfm = summary.get("dfm_output")

        st["status"] = "completed"
        st["updated_at"] = _now()
        st["message"] = "Completed"
        st["run_dir"] = str(run_dir)
        st["result_dfm"] = result_dfm
        _write_status(job_dir, st)
    except Exception as e:
        st = _read_status(job_dir)
        st["status"] = "failed"
        st["updated_at"] = _now()
        st["message"] = f"{type(e).__name__}: {e}"
        (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        _write_status(job_dir, st)


@app.get("/healthz")
def healthz():
    return {"ok": True}


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
    return JobStatus(**_read_status(job_dir))


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail_lines: int = 200):
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")

    st = _read_status(job_dir)
    run_dir = st.get("run_dir")
    if not run_dir:
        return PlainTextResponse("Run has not started yet.")

    log_file = Path(run_dir) / "autotrain.log"
    if not log_file.exists():
        return PlainTextResponse("No log yet.")

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
