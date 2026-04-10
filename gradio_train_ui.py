import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import gradio as gr
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parent
MAIN_PY = ROOT_DIR / "main.py"
RUNS_DIR = ROOT_DIR / "webui_runs"
MODEL_CHOICES = ["Model_SAEHD", "Model_Quick96", "Model_AMP"]
FACE_TYPES = ["whole_face", "full_face", "head", "half_face"]
DETECTORS = ["s3fd", "manual"]


def generate_presets() -> Dict[str, Dict[str, object]]:
    presets: Dict[str, Dict[str, object]] = {}
    for mode in ("fast", "balanced", "quality"):
        for backend in ("gpu", "cpu"):
            name = f"{mode}_{backend}"
            if mode == "fast":
                fps = 8
                train_minutes = 20
                image_size = 256
                model = "Model_Quick96"
            elif mode == "balanced":
                fps = 0
                train_minutes = 45
                image_size = 384
                model = "Model_SAEHD"
            else:
                fps = 0
                train_minutes = 90
                image_size = 512
                model = "Model_AMP"

            presets[name] = {
                "model_name": model,
                "face_type": "whole_face",
                "detector": "s3fd",
                "fps": fps,
                "image_size": image_size,
                "jpeg_quality": 90 if mode != "quality" else 95,
                "max_faces": 1,
                "cpu_only": backend == "cpu",
                "training_minutes": train_minutes,
                "silent_start": True,
                "no_preview": True,
            }
    return presets


PRESETS = generate_presets()
ITER_PATTERN = re.compile(r"\[(\d+)\]")
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")
ACTIVE_LOCK = threading.Lock()
ACTIVE_PROCESS: Optional[subprocess.Popen] = None
STOP_EVENT = threading.Event()
API_HOST = "127.0.0.1"
API_PORT = 7861
API_BASE = f"http://{API_HOST}:{API_PORT}"


class TrainRequest(BaseModel):
    src_video: Optional[str] = None
    src_images: Optional[List[str]] = None
    dst_video: Optional[str] = None
    dst_images: Optional[List[str]] = None
    preset_name: str
    model_name: str
    face_type: str
    detector: str
    fps: int
    image_size: int
    jpeg_quality: int
    max_faces: int
    force_gpu_idxs: str = ""
    cpu_only: bool = False
    training_minutes: int = 45
    max_iterations: int = 50000
    enable_early_stop: bool = True
    min_iter_for_early_stop: int = 20000
    patience_windows: int = 5
    window_size: int = 1000
    min_delta_percent: float = 0.2
    no_preview: bool = True
    silent_start: bool = True


class JobState:
    def __init__(self) -> None:
        self.status = "queued"
        self.logs = ""
        self.dfm_file: Optional[str] = None
        self.running = True
        self.error: Optional[str] = None


JOB_LOCK = threading.Lock()
JOBS: Dict[str, JobState] = {}
app_api = FastAPI(title="DeepFaceLab Training API")


def _normalize_model_arg(model_name: str) -> str:
    if model_name.startswith("Model_"):
        return model_name.replace("Model_", "", 1)
    return model_name


def _copy_files(files: Optional[List[str]], target_dir: Path) -> int:
    if not files:
        return 0
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fp in files:
        src = Path(fp)
        if src.exists():
            shutil.copy2(src, target_dir / src.name)
            copied += 1
    return copied


def _normalize_file_path(file_obj: Any) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return str(file_obj.name)
    return None


def _normalize_file_list(files_obj: Any) -> List[str]:
    if files_obj is None:
        return []
    if not isinstance(files_obj, list):
        files_obj = [files_obj]
    out: List[str] = []
    for item in files_obj:
        fp = _normalize_file_path(item)
        if fp:
            out.append(fp)
    return out


def _set_active_process(process: Optional[subprocess.Popen]) -> None:
    global ACTIVE_PROCESS
    with ACTIVE_LOCK:
        ACTIVE_PROCESS = process


def stop_active_training() -> str:
    STOP_EVENT.set()
    with ACTIVE_LOCK:
        process = ACTIVE_PROCESS
    if process is None:
        return "No active process to stop."
    try:
        process.terminate()
        return "Stop requested. Waiting for process to exit..."
    except Exception as exc:  # pragma: no cover
        return f"Stop request failed: {exc}"


def _run_command(
    cmd: List[str],
    cwd: Path,
    logs: List[str],
    log_file: Path,
    timeout_seconds: Optional[int] = None,
    on_line: Optional[Callable[[str], None]] = None,
    on_logs_updated: Optional[Callable[[str], None]] = None,
) -> Tuple[int, bool, bool]:
    logs.append("")
    logs.append(f"[CMD] {' '.join(cmd)}")
    logs.append(f"[CWD] {cwd}")
    log_file.write_text("\n".join(logs), encoding="utf-8")
    if on_logs_updated is not None:
        on_logs_updated("\n".join(logs))

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    _set_active_process(process)

    timed_out = False
    stopped_by_user = False
    started = time.time()

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        logs.append(line)
        if on_line is not None:
            on_line(line)
        log_file.write_text("\n".join(logs), encoding="utf-8")
        if on_logs_updated is not None:
            on_logs_updated("\n".join(logs))
        if STOP_EVENT.is_set():
            stopped_by_user = True
            logs.append("[WARN] Stop requested by user. Terminating process.")
            process.terminate()
            break
        if timeout_seconds is not None and (time.time() - started) >= timeout_seconds:
            timed_out = True
            logs.append(f"[WARN] Timeout reached after {timeout_seconds}s. Stopping process.")
            process.terminate()
            break

    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    finally:
        _set_active_process(None)
        log_file.write_text("\n".join(logs), encoding="utf-8")
        if on_logs_updated is not None:
            on_logs_updated("\n".join(logs))

    return process.returncode or 0, timed_out, stopped_by_user


def _find_latest_dfm(model_dir: Path) -> Optional[Path]:
    dfm_files = sorted(model_dir.glob("*.dfm"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dfm_files[0] if dfm_files else None


def _extract_iteration(log_line: str) -> Optional[int]:
    matches = ITER_PATTERN.findall(log_line)
    if not matches:
        return None
    try:
        return int(matches[0])
    except ValueError:
        return None


def _extract_loss_value(log_line: str) -> Optional[float]:
    if "[" not in log_line or "]" not in log_line:
        return None
    it = _extract_iteration(log_line)
    if it is None:
        return None
    tail = log_line.split("]", 1)[-1]
    values: List[float] = []
    for token in FLOAT_PATTERN.findall(tail):
        try:
            values.append(float(token))
        except ValueError:
            continue
    if not values:
        return None
    if len(values) >= 2:
        return (values[0] + values[1]) / 2.0
    return values[0]


def _format_seconds(seconds: float) -> str:
    sec = max(0, int(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _save_run_meta(run_dir: Path, payload: Dict[str, object]) -> None:
    (run_dir / "run_meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def list_run_ids() -> List[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [p.name for p in RUNS_DIR.iterdir() if p.is_dir()]
    runs.sort(reverse=True)
    return runs


def refresh_runs() -> gr.Dropdown:
    return gr.Dropdown.update(choices=list_run_ids())


def _toggle_input_panels(input_mode: str) -> Tuple[gr.File, gr.File]:
    video_visible = input_mode == "video"
    images_visible = input_mode == "images"
    return gr.File.update(visible=video_visible), gr.File.update(visible=images_visible)


def load_run_history(run_id: str) -> Tuple[str, str, Optional[str]]:
    if not run_id:
        return "No run selected.", "", None
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return "Run not found.", "", None
    log_path = run_dir / "run.log"
    meta_path = run_dir / "run_meta.json"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    meta_text = meta_path.read_text(encoding="utf-8") if meta_path.exists() else "{}"
    dfm = _find_latest_dfm(run_dir / "model")
    return f"Loaded run: {run_id}", meta_text + "\n\n" + log_text, str(dfm) if dfm else None


def _apply_preset(
    preset_name: str,
    model_name: str,
    face_type: str,
    detector: str,
    fps: int,
    image_size: int,
    jpeg_quality: int,
    max_faces: int,
    cpu_only: bool,
    training_minutes: int,
    no_preview: bool,
    silent_start: bool,
) -> Tuple[str, str, str, int, int, int, int, bool, int, bool, bool]:
    if preset_name == "custom":
        return (
            model_name,
            face_type,
            detector,
            fps,
            image_size,
            jpeg_quality,
            max_faces,
            cpu_only,
            training_minutes,
            no_preview,
            silent_start,
        )
    conf = PRESETS[preset_name]
    return (
        conf["model_name"],
        conf["face_type"],
        conf["detector"],
        int(conf["fps"]),
        int(conf["image_size"]),
        int(conf["jpeg_quality"]),
        int(conf["max_faces"]),
        bool(conf["cpu_only"]),
        int(conf["training_minutes"]),
        bool(conf["no_preview"]),
        bool(conf["silent_start"]),
    )


def run_training_pipeline(
    src_video: Any,
    src_images: Any,
    dst_video: Any,
    dst_images: Any,
    preset_name: str,
    model_name: str,
    face_type: str,
    detector: str,
    fps: int,
    image_size: int,
    jpeg_quality: int,
    max_faces: int,
    force_gpu_idxs: str,
    cpu_only: bool,
    training_minutes: int,
    max_iterations: int,
    enable_early_stop: bool,
    min_iter_for_early_stop: int,
    patience_windows: int,
    window_size: int,
    min_delta_percent: float,
    no_preview: bool,
    silent_start: bool,
    on_logs_updated: Optional[Callable[[str], None]] = None,
    on_status_updated: Optional[Callable[[str], None]] = None,
) -> Generator[Tuple[str, str, Optional[str]], None, None]:
    yield from _run_training_pipeline_full(
        src_video=src_video,
        src_images=src_images,
        dst_video=dst_video,
        dst_images=dst_images,
        preset_name=preset_name,
        model_name=model_name,
        face_type=face_type,
        detector=detector,
        fps=fps,
        image_size=image_size,
        jpeg_quality=jpeg_quality,
        max_faces=max_faces,
        force_gpu_idxs=force_gpu_idxs,
        cpu_only=cpu_only,
        training_minutes=training_minutes,
        max_iterations=max_iterations,
        no_preview=no_preview,
        silent_start=silent_start,
        on_logs_updated=on_logs_updated,
        on_status_updated=on_status_updated,
    )


def _run_training_pipeline_full(
    src_video: Any,
    src_images: Any,
    dst_video: Any,
    dst_images: Any,
    preset_name: str,
    model_name: str,
    face_type: str,
    detector: str,
    fps: int,
    image_size: int,
    jpeg_quality: int,
    max_faces: int,
    force_gpu_idxs: str,
    cpu_only: bool,
    training_minutes: int,
    max_iterations: int,
    no_preview: bool,
    silent_start: bool,
    on_logs_updated: Optional[Callable[[str], None]] = None,
    on_status_updated: Optional[Callable[[str], None]] = None,
) -> Generator[Tuple[str, str, Optional[str]], None, None]:
    progress = gr.Progress(track_tqdm=False)
    logs: List[str] = []
    status = "Starting..."
    output_file: Optional[str] = None
    STOP_EVENT.clear()

    src_video_path = _normalize_file_path(src_video)
    dst_video_path = _normalize_file_path(dst_video)
    src_images_list = _normalize_file_list(src_images)
    dst_images_list = _normalize_file_list(dst_images)

    (
        model_name,
        face_type,
        detector,
        fps,
        image_size,
        jpeg_quality,
        max_faces,
        cpu_only,
        training_minutes,
        no_preview,
        silent_start,
    ) = _apply_preset(
        preset_name, model_name, face_type, detector, fps, image_size, jpeg_quality, max_faces, cpu_only, training_minutes, no_preview, silent_start
    )
    model_arg = _normalize_model_arg(model_name)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
    work_dir = RUNS_DIR / run_id
    src_dir = work_dir / "data_src"
    dst_dir = work_dir / "data_dst"
    src_aligned = src_dir / "aligned"
    dst_aligned = dst_dir / "aligned"
    model_dir = work_dir / "model"
    for p in (src_dir, dst_dir, src_aligned, dst_aligned, model_dir):
        p.mkdir(parents=True, exist_ok=True)
    log_file = work_dir / "run.log"

    logs.extend([
        f"[INFO] Run ID: {run_id}",
        f"[INFO] Work dir: {work_dir}",
        f"[INFO] Preset: {preset_name}",
        f"[INFO] Model: {model_name}",
        f"[INFO] Model arg: {model_arg}",
        f"[INFO] CPU only: {cpu_only}",
        f"[INFO] Max iterations: {max_iterations}",
        f"[INFO] Training timeout minutes: {training_minutes}",
    ])
    log_file.write_text("\n".join(logs), encoding="utf-8")
    if on_logs_updated:
        on_logs_updated("\n".join(logs))
    yield status, "\n".join(logs), output_file

    if not src_video_path and not src_images_list:
        logs.append("[ERROR] Missing SRC input.")
        yield "Failed", "\n".join(logs), None
        return
    if not dst_video_path and not dst_images_list:
        logs.append("[ERROR] Missing DST input.")
        yield "Failed", "\n".join(logs), None
        return

    src_video_work_path = None
    dst_video_work_path = None
    if src_video_path:
        t = src_dir / Path(src_video_path).name
        shutil.copy2(src_video_path, t)
        src_video_work_path = str(t)
    else:
        logs.append(f"[INFO] Copied {_copy_files(src_images_list, src_dir)} SRC images.")
    if dst_video_path:
        t = dst_dir / Path(dst_video_path).name
        shutil.copy2(dst_video_path, t)
        dst_video_work_path = str(t)
    else:
        logs.append(f"[INFO] Copied {_copy_files(dst_images_list, dst_dir)} DST images.")

    python_cmd = [sys.executable, "main.py"]
    if cpu_only:
        device_args = ["--cpu-only"]
    else:
        device_args = ["--force-gpu-idxs", force_gpu_idxs.strip() or "0"]

    def _run_or_fail(stage: str, cmd: List[str], prog: float, timeout_seconds: Optional[int] = None) -> bool:
        nonlocal status
        progress(prog, desc=stage)
        status = stage
        if on_status_updated:
            on_status_updated(status)
        code, timed_out, stopped = _run_command(
            cmd, ROOT_DIR, logs, log_file, timeout_seconds=timeout_seconds, on_logs_updated=on_logs_updated
        )
        if stopped:
            logs.append("[WARN] Stopped by user.")
            return False
        if timed_out:
            status = f"Timed out at {stage}"
            logs.append(f"[ERROR] Stage timed out: {stage}")
            return False
        if code != 0:
            status = f"Failed at {stage}"
            logs.append(f"[ERROR] Stage failed: {stage}")
            if on_status_updated:
                on_status_updated(status)
            return False
        return True

    def _has_aligned_faces(path: Path) -> bool:
        return any(path.glob("*.jpg")) or any(path.glob("*.png"))

    def _extract_faces_with_retry(side_label: str, input_dir: Path, aligned_dir: Path, prog: float) -> bool:
        nonlocal status
        attempts = [
            {"detector": detector, "image_size": image_size, "face_type": face_type},
            {"detector": "s3fd", "image_size": max(256, min(image_size, 320)), "face_type": "full_face"},
            {"detector": "s3fd", "image_size": 256, "face_type": "whole_face"},
        ]

        for idx, cfg in enumerate(attempts, start=1):
            if _has_aligned_faces(aligned_dir):
                return True
            logs.append(
                f"[INFO] {side_label} extract attempt {idx}/{len(attempts)}: "
                f"detector={cfg['detector']}, image_size={cfg['image_size']}, face_type={cfg['face_type']}"
            )
            if not _run_or_fail(
                f"Extracting {side_label} faces (attempt {idx})",
                python_cmd
                + [
                    "extract",
                    "--detector",
                    str(cfg["detector"]),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(aligned_dir),
                    "--no-output-debug",
                    "--face-type",
                    str(cfg["face_type"]),
                    "--max-faces-from-image",
                    str(max_faces),
                    "--image-size",
                    str(cfg["image_size"]),
                    "--jpeg-quality",
                    str(jpeg_quality),
                ]
                + device_args,
                prog,
            ):
                return False
            if _has_aligned_faces(aligned_dir):
                logs.append(f"[INFO] {side_label} aligned faces found after attempt {idx}.")
                return True
            logs.append(f"[WARN] {side_label} still empty after attempt {idx}.")

        status = f"Failed: {side_label} aligned is empty"
        logs.append(f"[ERROR] {side_label} aligned folder has no extracted faces after retries.")
        if on_status_updated:
            on_status_updated(status)
        return False

    if src_video_work_path and not _run_or_fail("Extracting SRC frames from video", python_cmd + ["videoed", "extract-video", "--input-file", src_video_work_path, "--output-dir", str(src_dir), "--output-ext", "png", "--fps", str(fps)], 0.10):
        yield status, "\n".join(logs), None
        return
    if dst_video_work_path and not _run_or_fail("Extracting DST frames from video", python_cmd + ["videoed", "extract-video", "--input-file", dst_video_work_path, "--output-dir", str(dst_dir), "--output-ext", "png", "--fps", str(fps)], 0.18):
        yield status, "\n".join(logs), None
        return
    if not _extract_faces_with_retry("SRC", src_dir, src_aligned, 0.30):
        yield status, "\n".join(logs), None
        return
    if not _extract_faces_with_retry("DST", dst_dir, dst_aligned, 0.45):
        yield status, "\n".join(logs), None
        return

    train_cmd = python_cmd + [
        "train",
        "--training-data-src-dir",
        str(src_aligned),
        "--training-data-dst-dir",
        str(dst_aligned),
        "--model-dir",
        str(model_dir),
        "--model",
        model_arg,
        "--force-model-name",
        "webui_model",
    ] + device_args
    if no_preview:
        train_cmd.append("--no-preview")
    if silent_start:
        train_cmd.append("--silent-start")
    if max_iterations > 0:
        train_cmd += ["--execute-program", str(max_iterations), "self", "shutdown"]
    if not _run_or_fail("Training model", train_cmd, 0.70, timeout_seconds=max(60, training_minutes * 60)):
        yield status, "\n".join(logs), None
        return

    if not _run_or_fail("Exporting .dfm model", python_cmd + ["exportdfm", "--model-dir", str(model_dir), "--model", model_arg], 0.90):
        yield status, "\n".join(logs), None
        return

    dfm = _find_latest_dfm(model_dir)
    if not dfm:
        yield "Completed with error", "\n".join(logs + ["[ERROR] No .dfm file found."]), None
        return
    output_file = str(dfm)
    logs.append(f"[OK] DFM ready: {output_file}")
    progress(1.0, desc="Completed")
    status = "Completed successfully"
    if on_status_updated:
        on_status_updated(status)
    yield status, "\n".join(logs), output_file


def _run_job_thread(job_id: str, req: TrainRequest) -> None:
    try:
        def _on_logs_updated(log_text: str) -> None:
            with JOB_LOCK:
                if job_id in JOBS:
                    JOBS[job_id].logs = log_text
        def _on_status_updated(status_text: str) -> None:
            with JOB_LOCK:
                if job_id in JOBS:
                    JOBS[job_id].status = status_text

        for status, logs, dfm in run_training_pipeline(
            req.src_video,
            req.src_images,
            req.dst_video,
            req.dst_images,
            req.preset_name,
            req.model_name,
            req.face_type,
            req.detector,
            req.fps,
            req.image_size,
            req.jpeg_quality,
            req.max_faces,
            req.force_gpu_idxs,
            req.cpu_only,
            req.training_minutes,
            req.max_iterations,
            req.enable_early_stop,
            req.min_iter_for_early_stop,
            req.patience_windows,
            req.window_size,
            req.min_delta_percent,
            req.no_preview,
            req.silent_start,
            on_logs_updated=_on_logs_updated,
            on_status_updated=_on_status_updated,
        ):
            with JOB_LOCK:
                if job_id in JOBS:
                    JOBS[job_id].status = status
                    JOBS[job_id].logs = logs
                    JOBS[job_id].dfm_file = dfm
        with JOB_LOCK:
            if job_id in JOBS:
                JOBS[job_id].running = False
                if JOBS[job_id].status.lower().startswith("failed"):
                    JOBS[job_id].error = JOBS[job_id].status
    except Exception as exc:
        with JOB_LOCK:
            if job_id in JOBS:
                JOBS[job_id].running = False
                JOBS[job_id].status = "Failed"
                JOBS[job_id].error = str(exc)
                JOBS[job_id].logs = (JOBS[job_id].logs + f"\n[API_ERROR] {exc}").strip()


@app_api.post("/jobs/start")
def api_start_job(req: TrainRequest) -> Dict[str, str]:
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
    with JOB_LOCK:
        JOBS[job_id] = JobState()
    thread = threading.Thread(target=_run_job_thread, args=(job_id, req), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app_api.get("/jobs/{job_id}")
def api_get_job(job_id: str) -> Dict[str, Any]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job_id,
            "status": job.status,
            "running": job.running,
            "dfm_file": job.dfm_file,
            "error": job.error,
        }


@app_api.get("/jobs/{job_id}/logs")
def api_get_logs(job_id: str) -> Dict[str, str]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"logs": job.logs}


@app_api.post("/jobs/{job_id}/stop")
def api_stop_job(job_id: str) -> Dict[str, str]:
    with JOB_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
    msg = stop_active_training()
    return {"message": msg}


@app_api.get("/jobs/{job_id}/dfm")
def api_get_dfm(job_id: str) -> FileResponse:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        dfm_path = job.dfm_file
    if not dfm_path or not Path(dfm_path).exists():
        raise HTTPException(status_code=404, detail="DFM not ready")
    return FileResponse(dfm_path, media_type="application/octet-stream", filename=Path(dfm_path).name)


def _start_api_server() -> None:
    config = uvicorn.Config(app_api, host=API_HOST, port=API_PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def _start_job_via_api(
    src_video: Any,
    src_images: Any,
    dst_video: Any,
    dst_images: Any,
    preset_name: str,
    model_name: str,
    face_type: str,
    detector: str,
    fps: int,
    image_size: int,
    jpeg_quality: int,
    max_faces: int,
    force_gpu_idxs: str,
    cpu_only: bool,
    training_minutes: int,
    max_iterations: int,
    enable_early_stop: bool,
    min_iter_for_early_stop: int,
    patience_windows: int,
    window_size: int,
    min_delta_percent: float,
    no_preview: bool,
    silent_start: bool,
) -> Generator[Tuple[str, str, Optional[str], str], None, None]:
    payload = {
        "src_video": _normalize_file_path(src_video),
        "src_images": _normalize_file_list(src_images),
        "dst_video": _normalize_file_path(dst_video),
        "dst_images": _normalize_file_list(dst_images),
        "preset_name": preset_name,
        "model_name": model_name,
        "face_type": face_type,
        "detector": detector,
        "fps": fps,
        "image_size": image_size,
        "jpeg_quality": jpeg_quality,
        "max_faces": max_faces,
        "force_gpu_idxs": force_gpu_idxs,
        "cpu_only": cpu_only,
        "training_minutes": training_minutes,
        "max_iterations": max_iterations,
        "enable_early_stop": enable_early_stop,
        "min_iter_for_early_stop": min_iter_for_early_stop,
        "patience_windows": patience_windows,
        "window_size": window_size,
        "min_delta_percent": min_delta_percent,
        "no_preview": no_preview,
        "silent_start": silent_start,
    }
    resp = requests.post(f"{API_BASE}/jobs/start", json=payload, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    status = "Job started"
    logs = f"[API] job_id={job_id}"
    dfm_file: Optional[str] = None
    yield status, logs, dfm_file, job_id

    while True:
        time.sleep(2)
        state_resp = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=30)
        logs_resp = requests.get(f"{API_BASE}/jobs/{job_id}/logs", timeout=30)
        state_resp.raise_for_status()
        logs_resp.raise_for_status()
        state = state_resp.json()
        logs = logs_resp.json().get("logs", "")
        status = state.get("status", "Unknown")
        dfm_file = state.get("dfm_file")
        yield status, logs, dfm_file, job_id
        if not state.get("running", False):
            break


def _stop_job_via_api(job_id: str) -> str:
    if not job_id:
        return "No active job id."
    resp = requests.post(f"{API_BASE}/jobs/{job_id}/stop", timeout=15)
    if resp.status_code >= 400:
        return f"Stop failed: {resp.text}"
    return resp.json().get("message", "Stop requested.")

    progress(0.02, desc="Preparing input data")
    status = "Preparing input data"
    if src_video_path:
        src_video_target = src_dir / Path(src_video_path).name
        shutil.copy2(src_video_path, src_video_target)
        src_video_work_path = str(src_video_target)
    else:
        copied = _copy_files(src_image_paths, src_dir)
        logs.append(f"[INFO] Copied {copied} SRC images.")

    if dst_video_path:
        dst_video_target = dst_dir / Path(dst_video_path).name
        shutil.copy2(dst_video_path, dst_video_target)
        dst_video_work_path = str(dst_video_target)
    else:
        copied = _copy_files(dst_image_paths, dst_dir)
        logs.append(f"[INFO] Copied {copied} DST images.")

    if not src_video_path and not src_image_paths:
        logs.append("[ERROR] Missing SRC input. Upload a SRC video or SRC image files.")
        yield "Failed", "\n".join(logs), None
        return
    if not dst_video_path and not dst_image_paths:
        logs.append("[ERROR] Missing DST input. Upload a DST video or DST image files.")
        yield "Failed", "\n".join(logs), None
        return

    yield status, "\n".join(logs), output_file

    python_cmd = [sys.executable, "main.py"]
    gpu_args: List[str] = []
    normalized_gpu_idxs = force_gpu_idxs.strip()
    if cpu_only:
        gpu_args = ["--cpu-only"]
        logs.append("[INFO] Running in CPU-only mode.")
    else:
        if not normalized_gpu_idxs:
            normalized_gpu_idxs = "0"
            logs.append("[INFO] Force GPU idxs is empty, defaulting to GPU 0.")
        gpu_args = ["--force-gpu-idxs", normalized_gpu_idxs]
        logs.append(f"[INFO] Running with forced GPU idxs: {normalized_gpu_idxs}")

    if src_video_work_path:
        progress(0.10, desc="Extracting SRC frames from video")
        status = "Extracting SRC frames from video"
        cmd = python_cmd + [
            "videoed",
            "extract-video",
            "--input-file",
            src_video_work_path,
            "--output-dir",
            str(src_dir),
            "--output-ext",
            "png",
            "--fps",
            str(fps),
        ]
        code, _, stopped = _run_command(cmd, ROOT_DIR, logs, log_file, on_logs_updated=on_logs_updated)
        yield status, "\n".join(logs), output_file
        if stopped:
            yield "Stopped by user", "\n".join(logs), None
            return
        if code != 0:
            yield "Failed at SRC video extraction", "\n".join(logs), None
            return

    if dst_video_work_path:
        progress(0.18, desc="Extracting DST frames from video")
        status = "Extracting DST frames from video"
        cmd = python_cmd + [
            "videoed",
            "extract-video",
            "--input-file",
            dst_video_work_path,
            "--output-dir",
            str(dst_dir),
            "--output-ext",
            "png",
            "--fps",
            str(fps),
        ]
        code, _, stopped = _run_command(cmd, ROOT_DIR, logs, log_file, on_logs_updated=on_logs_updated)
        yield status, "\n".join(logs), output_file
        if stopped:
            yield "Stopped by user", "\n".join(logs), None
            return
        if code != 0:
            yield "Failed at DST video extraction", "\n".join(logs), None
            return

    progress(0.30, desc="Extracting SRC faces")
    status = "Extracting SRC faces"
    cmd = python_cmd + [
        "extract",
        "--detector",
        detector,
        "--input-dir",
        str(src_dir),
        "--output-dir",
        str(src_aligned),
        "--no-output-debug",
        "--face-type",
        face_type,
        "--max-faces-from-image",
        str(max_faces),
        "--image-size",
        str(image_size),
        "--jpeg-quality",
        str(jpeg_quality),
    ] + gpu_args
    code, _, stopped = _run_command(cmd, ROOT_DIR, logs, log_file, on_logs_updated=on_logs_updated)
    yield status, "\n".join(logs), output_file
    if stopped:
        yield "Stopped by user", "\n".join(logs), None
        return
    if code != 0:
        yield "Failed at SRC face extraction", "\n".join(logs), None
        return

    progress(0.45, desc="Extracting DST faces")
    status = "Extracting DST faces"
    cmd = python_cmd + [
        "extract",
        "--detector",
        detector,
        "--input-dir",
        str(dst_dir),
        "--output-dir",
        str(dst_aligned),
        "--no-output-debug",
        "--face-type",
        face_type,
        "--max-faces-from-image",
        str(max_faces),
        "--image-size",
        str(image_size),
        "--jpeg-quality",
        str(jpeg_quality),
    ] + gpu_args
    code, _, stopped = _run_command(cmd, ROOT_DIR, logs, log_file, on_logs_updated=on_logs_updated)
    yield status, "\n".join(logs), output_file
    if stopped:
        yield "Stopped by user", "\n".join(logs), None
        return
    if code != 0:
        yield "Failed at DST face extraction", "\n".join(logs), None
        return

    progress(0.60, desc="Training model")
    status = "Training model"
    train_cmd = python_cmd + [
        "train",
        "--training-data-src-dir",
        str(src_aligned),
        "--training-data-dst-dir",
        str(dst_aligned),
        "--model-dir",
        str(model_dir),
        "--model",
        model_name,
    ] + gpu_args
    if no_preview:
        train_cmd.append("--no-preview")
    if silent_start:
        train_cmd.append("--silent-start")
    if max_iterations > 0:
        train_cmd += ["--execute-program", str(max_iterations), "self", "shutdown"]

    train_timeout_seconds = max(1, training_minutes) * 60
    current_iter = 0
    last_loss: Optional[float] = None
    window_best_loss: Optional[float] = None
    stale_windows = 0
    current_window_start = 0
    stop_reason = "completed"
    train_started_at = time.time()
    last_status_emit_at = 0.0
    status_text = "Training model"

    def _on_train_line(line: str) -> None:
        nonlocal current_iter, last_loss, window_best_loss, stale_windows, current_window_start, stop_reason
        nonlocal last_status_emit_at, status_text
        it = _extract_iteration(line)
        if it is not None:
            current_iter = max(current_iter, it)

        now = time.time()
        elapsed = max(1e-6, now - train_started_at)
        speed = current_iter / elapsed if current_iter > 0 else 0.0
        eta_text = "N/A"
        if max_iterations > 0 and speed > 0:
            remain_iter = max(0, max_iterations - current_iter)
            eta_text = _format_seconds(remain_iter / speed)

        if (now - last_status_emit_at) >= 2.0 and current_iter > 0:
            target_text = str(max_iterations) if max_iterations > 0 else "unlimited"
            status_text = (
                f"Training model | iter={current_iter}/{target_text} | "
                f"speed={speed:.2f} it/s | ETA={eta_text}"
            )
            logs.append(f"[ETA] {status_text}")
            last_status_emit_at = now

        loss_val = _extract_loss_value(line)
        if loss_val is None:
            return
        last_loss = loss_val
        if window_best_loss is None:
            window_best_loss = loss_val
        else:
            window_best_loss = min(window_best_loss, loss_val)

        if current_window_start == 0:
            current_window_start = current_iter
            return

        if (current_iter - current_window_start) < max(1, window_size):
            return

        if enable_early_stop and current_iter >= max(0, min_iter_for_early_stop):
            rel_improve = 0.0
            if window_best_loss and loss_val <= window_best_loss:
                rel_improve = ((window_best_loss - loss_val) / max(window_best_loss, 1e-8)) * 100.0
            if rel_improve < max(0.0, min_delta_percent):
                stale_windows += 1
                logs.append(
                    f"[EARLY_STOP] stale window {stale_windows}/{patience_windows} "
                    f"(iter={current_iter}, improve={rel_improve:.4f}%)"
                )
            else:
                stale_windows = 0
                logs.append(
                    f"[EARLY_STOP] improvement detected "
                    f"(iter={current_iter}, improve={rel_improve:.4f}%)"
                )

            if stale_windows >= max(1, patience_windows):
                logs.append("[EARLY_STOP] Plateau detected. Requesting graceful stop.")
                stop_reason = "early_stop_plateau"
                STOP_EVENT.set()

        current_window_start = current_iter
        window_best_loss = loss_val

    train_code, timed_out, stopped = _run_command(
        train_cmd,
        ROOT_DIR,
        logs,
        log_file,
        timeout_seconds=train_timeout_seconds,
        on_line=_on_train_line,
        on_logs_updated=on_logs_updated,
    )
    status = status_text
    logs.append(f"[INFO] Training return code: {train_code}")
    logs.append(f"[INFO] Last seen iteration: {current_iter}")
    elapsed_total = max(1e-6, time.time() - train_started_at)
    avg_speed = current_iter / elapsed_total if current_iter > 0 else 0.0
    logs.append(f"[INFO] Avg training speed: {avg_speed:.2f} it/s")
    if last_loss is not None:
        logs.append(f"[INFO] Last seen loss: {last_loss:.6f}")
    if stopped:
        if stop_reason == "early_stop_plateau":
            logs.append("[INFO] Training stopped due to plateau early-stop.")
        else:
            stop_reason = "manual_stop"
            yield "Stopped during training", "\n".join(logs), None
            return
    if timed_out:
        stop_reason = "timeout"
        logs.append("[INFO] Training was stopped by timeout. Continuing with latest checkpoint.")

    yield status, "\n".join(logs), output_file
    if train_code != 0 and not timed_out:
        yield "Failed at training", "\n".join(logs), None
        return

    progress(0.85, desc="Exporting DFM")
    status = "Exporting .dfm model"
    export_cmd = python_cmd + [
        "exportdfm",
        "--model-dir",
        str(model_dir),
        "--model",
        model_name,
    ]
    code, _, stopped = _run_command(export_cmd, ROOT_DIR, logs, log_file, on_logs_updated=on_logs_updated)
    yield status, "\n".join(logs), output_file
    if stopped:
        yield "Stopped by user", "\n".join(logs), None
        return
    if code != 0:
        yield "Failed at exportdfm", "\n".join(logs), None
        return

    progress(1.0, desc="Completed")
    dfm = _find_latest_dfm(model_dir)
    if dfm is None:
        logs.append("[ERROR] No .dfm file found in model directory.")
        yield "Completed with error", "\n".join(logs), None
        return

    output_file = str(dfm)
    logs.append(f"[OK] DFM ready: {output_file}")
    status = "Completed successfully"
    _save_run_meta(
        work_dir,
        {
            "run_id": run_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "preset": preset_name,
            "model": model_name,
            "face_type": face_type,
            "detector": detector,
            "cpu_only": cpu_only,
            "max_iterations": max_iterations,
            "training_minutes": training_minutes,
            "last_seen_iteration": current_iter,
            "last_seen_loss": last_loss,
            "avg_training_speed_iter_per_sec": avg_speed,
            "stop_reason": stop_reason,
            "dfm_file": output_file,
            "status": status,
        },
    )
    log_file.write_text("\n".join(logs), encoding="utf-8")
    if on_logs_updated is not None:
        on_logs_updated("\n".join(logs))
    yield status, "\n".join(logs), output_file


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DeepFaceLab Trainer WebUI") as demo:
        with gr.Tab("Train"):
            gr.Markdown(
                "# DeepFaceLab Trainer WebUI\n"
                "Upload SRC/DST data as video or image sets, choose preset/settings, train, and download `.dfm`."
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("## SRC input")
                    src_mode = gr.Radio(
                        choices=["video", "images"],
                        value="video",
                        label="SRC mode",
                    )
                    src_video = gr.File(
                        label="SRC video (optional, overrides SRC images)",
                        file_types=["video"],
                        type="file",
                        visible=True,
                    )
                    src_images = gr.File(
                        label="SRC images (multiple)",
                        file_types=["image"],
                        file_count="multiple",
                        type="file",
                        visible=False,
                    )
                with gr.Column():
                    gr.Markdown("## DST input")
                    dst_mode = gr.Radio(
                        choices=["video", "images"],
                        value="video",
                        label="DST mode",
                    )
                    dst_video = gr.File(
                        label="DST video (optional, overrides DST images)",
                        file_types=["video"],
                        type="file",
                        visible=True,
                    )
                    dst_images = gr.File(
                        label="DST images (multiple)",
                        file_types=["image"],
                        file_count="multiple",
                        type="file",
                        visible=False,
                    )

            gr.Markdown("## Preset and params")
            with gr.Row():
                preset_name = gr.Dropdown(
                    choices=["custom"] + list(PRESETS.keys()),
                    value="balanced_gpu",
                    label="Preset",
                )
                model_name = gr.Dropdown(choices=MODEL_CHOICES, value="Model_SAEHD", label="Model")
                detector = gr.Dropdown(choices=DETECTORS, value="s3fd", label="Detector")
                face_type = gr.Dropdown(choices=FACE_TYPES, value="whole_face", label="Face type")

            with gr.Row():
                fps = gr.Slider(0, 30, value=0, step=1, label="Video extract FPS (0=full)")
                image_size = gr.Slider(128, 1024, value=384, step=32, label="Extract image size")
                jpeg_quality = gr.Slider(50, 100, value=90, step=1, label="JPEG quality")
                max_faces = gr.Slider(1, 5, value=1, step=1, label="Max faces per image")

            with gr.Row():
                force_gpu_idxs = gr.Textbox(
                    label="Force GPU idxs (optional, ex: 0 or 0,1)",
                    value="",
                )
                cpu_only = gr.Checkbox(label="CPU only", value=False)
                training_minutes = gr.Slider(
                    5, 600, value=45, step=5, label="Training timeout (minutes)"
                )
                max_iterations = gr.Slider(
                    0, 500000, value=50000, step=500, label="Target max iterations (0 = no auto-stop)"
                )
                enable_early_stop = gr.Checkbox(label="Enable smart early-stop (plateau)", value=True)
                min_iter_for_early_stop = gr.Slider(
                    1000, 500000, value=20000, step=500, label="Early-stop min iterations"
                )
                patience_windows = gr.Slider(
                    1, 20, value=5, step=1, label="Early-stop patience windows"
                )
                window_size = gr.Slider(
                    100, 10000, value=1000, step=100, label="Early-stop window size (iterations)"
                )
                min_delta_percent = gr.Slider(
                    0.01, 5.0, value=0.2, step=0.01, label="Early-stop min delta (%)"
                )
                no_preview = gr.Checkbox(label="No preview", value=True)
                silent_start = gr.Checkbox(label="Silent start", value=True)

            with gr.Row():
                start_btn = gr.Button("Train + Export .dfm", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop")
            status_box = gr.Textbox(label="Status")
            logs_box = gr.Textbox(label="Detailed logs", lines=20, max_lines=20)
            dfm_output = gr.File(label="Download trained .dfm")
            job_id_box = gr.Textbox(label="Job ID", interactive=False)

            preset_name.change(
                fn=_apply_preset,
                inputs=[
                    preset_name,
                    model_name,
                    face_type,
                    detector,
                    fps,
                    image_size,
                    jpeg_quality,
                    max_faces,
                    cpu_only,
                    training_minutes,
                    no_preview,
                    silent_start,
                ],
                outputs=[
                    model_name,
                    face_type,
                    detector,
                    fps,
                    image_size,
                    jpeg_quality,
                    max_faces,
                    cpu_only,
                    training_minutes,
                    no_preview,
                    silent_start,
                ],
            )

            start_btn.click(
                fn=_start_job_via_api,
                inputs=[
                    src_video,
                    src_images,
                    dst_video,
                    dst_images,
                    preset_name,
                    model_name,
                    face_type,
                    detector,
                    fps,
                    image_size,
                    jpeg_quality,
                    max_faces,
                    force_gpu_idxs,
                    cpu_only,
                    training_minutes,
                    max_iterations,
                    enable_early_stop,
                    min_iter_for_early_stop,
                    patience_windows,
                    window_size,
                    min_delta_percent,
                    no_preview,
                    silent_start,
                ],
                outputs=[status_box, logs_box, dfm_output, job_id_box],
            )
            stop_btn.click(fn=_stop_job_via_api, inputs=[job_id_box], outputs=[status_box])
            src_mode.change(fn=_toggle_input_panels, inputs=[src_mode], outputs=[src_video, src_images])
            dst_mode.change(fn=_toggle_input_panels, inputs=[dst_mode], outputs=[dst_video, dst_images])

        with gr.Tab("Run History"):
            gr.Markdown("## Previous runs")
            with gr.Row():
                run_selector = gr.Dropdown(choices=list_run_ids(), label="Run ID")
                refresh_btn = gr.Button("Refresh list")
                load_btn = gr.Button("Load selected run")
            history_status = gr.Textbox(label="History status")
            history_logs = gr.Textbox(label="Run meta + logs", lines=22, max_lines=22)
            history_dfm = gr.File(label="Download .dfm from selected run")
            refresh_btn.click(fn=refresh_runs, outputs=[run_selector])
            load_btn.click(fn=load_run_history, inputs=[run_selector], outputs=[history_status, history_logs, history_dfm])

    return demo


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    api_thread = threading.Thread(target=_start_api_server, daemon=True)
    api_thread.start()
    time.sleep(1.0)
    app = build_ui()
    app.queue(concurrency_count=1).launch(server_name="localhost", server_port=7860)
