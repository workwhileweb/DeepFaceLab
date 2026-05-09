import argparse
import contextlib
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.interact import interact as io
from core.leras import nn
from mainscripts import ExportDFM, Extractor, FacesetEnhancer, VideoEd, XSegUtil
import models


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


@dataclass
class AutoTrainConfig:
    src_video: Path
    dst_video: Path
    xseg_model_dir: Path
    workdir: Path
    model_name: str
    model_alias: str
    face_type: str
    detector: str
    extract_fps: int
    extract_image_size: int
    extract_jpeg_quality: int
    max_faces_from_image: int
    force_gpu_idxs: List[int]
    cpu_only: bool
    drop_ratio: float
    min_keep_ratio: float
    train_max_hours: float
    plateau_hours: float
    backup_minutes: int
    save_minutes: int
    loss_window_iters: int
    min_improve_delta: float
    best_criterion: str
    preset: str
    enable_xseg: bool
    enable_faceset_enhance: bool
    preprocess_parallel: bool
    parallel_align: bool
    parallel_enhance: bool
    train_batch_size: int
    min_iterations: int
    min_src_faces: int
    min_dst_faces: int


class TeeLogger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def dump_json(self, path: Path, obj: Dict) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)


_AUTO_IO_LOGGER: TeeLogger | None = None


def _auto_log(msg: str) -> None:
    if _AUTO_IO_LOGGER is not None:
        _AUTO_IO_LOGGER.log(msg)


def _auto_input(prompt="", *args, **kwargs):
    _auto_log(f"[auto-input] {prompt}")
    return ""


def _auto_input_str(prompt="", default_value="", *args, **kwargs):
    _auto_log(f"[auto-input-str] {prompt} -> {default_value}")
    return default_value


def _auto_input_int(prompt="", default_value=0, *args, **kwargs):
    value = default_value
    p = str(prompt).lower()
    # Allow forcing bigger batches to better utilize stronger GPUs.
    if "batch_size" in p:
        value = max(1, env_int("AUTOTRAIN_BATCH_SIZE", int(default_value) if default_value else 8))
    _auto_log(f"[auto-input-int] {prompt} -> {value}")
    return value


def _auto_input_number(prompt="", default_value=0.0, *args, **kwargs):
    _auto_log(f"[auto-input-number] {prompt} -> {default_value}")
    return default_value


def _auto_input_bool(prompt="", default_value=False, *args, **kwargs):
    _auto_log(f"[auto-input-bool] {prompt} -> {default_value}")
    return default_value


@contextlib.contextmanager
def patched_io_defaults(logger: TeeLogger):
    """Force all interactive prompts to use defaults."""
    attrs = ["input", "input_str", "input_int", "input_number", "input_bool"]
    saved = {a: getattr(io, a) for a in attrs}

    global _AUTO_IO_LOGGER
    _AUTO_IO_LOGGER = logger
    io.input = _auto_input
    io.input_str = _auto_input_str
    io.input_int = _auto_input_int
    io.input_number = _auto_input_number
    io.input_bool = _auto_input_bool
    try:
        yield
    finally:
        _AUTO_IO_LOGGER = None
        for a, v in saved.items():
            setattr(io, a, v)


def compute_image_metrics(img_bgr: np.ndarray, face_detector: cv2.CascadeClassifier) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(edges.mean() / 255.0)
    brightness = float(gray.mean())
    saturation = float(hsv[:, :, 1].mean())

    detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    face_found = 1.0 if len(detected) > 0 else 0.0
    face_cov = 0.0
    if len(detected) > 0:
        areas = [w * h for (_, _, w, h) in detected]
        face_cov = float(max(areas) / (gray.shape[0] * gray.shape[1]))

    # Tone and edge regularization scores.
    exp_score = float(max(0.0, 1.0 - abs(brightness - 127.0) / 127.0))
    sat_score = float(max(0.0, 1.0 - abs(saturation - 96.0) / 96.0))
    edge_score = float(min(1.0, edge_ratio / 0.15))

    return {
        "blur": blur,
        "edge_ratio": edge_ratio,
        "brightness": brightness,
        "saturation": saturation,
        "face_found": face_found,
        "face_cov": face_cov,
        "exp_score": exp_score,
        "sat_score": sat_score,
        "edge_score": edge_score,
    }


def norm(v: float, values: List[float]) -> float:
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return 0.5
    return (v - lo) / (hi - lo)


def filter_faceset(aligned_dir: Path, cfg: AutoTrainConfig, logger: TeeLogger, report_path: Path) -> Dict:
    img_paths = sorted([p for p in aligned_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not img_paths:
        return {"kept": 0, "rejected": 0, "note": "no aligned images"}

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    metrics = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = compute_image_metrics(img, detector)
        m["path"] = str(p)
        metrics.append(m)

    blur_vals = [m["blur"] for m in metrics]
    face_cov_vals = [m["face_cov"] for m in metrics]
    scored = []
    for m in metrics:
        score = (
            0.45 * norm(m["blur"], blur_vals)
            + 0.20 * m["edge_score"]
            + 0.15 * norm(m["face_cov"], face_cov_vals)
            + 0.10 * m["exp_score"]
            + 0.10 * m["sat_score"]
        )
        if m["face_found"] < 0.5:
            score *= 0.7
        mm = dict(m)
        mm["score"] = score
        scored.append(mm)

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    keep_n = max(int(len(scored) * cfg.min_keep_ratio), int(len(scored) * (1.0 - cfg.drop_ratio)))
    keep_n = max(1, min(keep_n, len(scored)))
    keep = scored[:keep_n]
    reject = scored[keep_n:]

    rejected_dir = aligned_dir / "_rejected_autotrain"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for item in reject:
        src = Path(item["path"])
        dst = rejected_dir / src.name
        if src.exists():
            shutil.move(str(src), str(dst))
            item["moved_to"] = str(dst)

    with (report_path.parent / f"{report_path.stem}.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["path", "score", "blur", "edge_ratio", "brightness", "saturation", "face_found", "face_cov"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in scored:
            writer.writerow({k: item.get(k) for k in fieldnames})

    report = {
        "input_total": len(metrics),
        "kept": len(keep),
        "rejected": len(reject),
        "drop_ratio_effective": (len(reject) / max(1, len(metrics))),
        "kept_examples": [k["path"] for k in keep[:10]],
        "rejected_examples": [r.get("moved_to", r["path"]) for r in reject[:10]],
    }
    logger.dump_json(report_path, report)
    logger.log(
        f"[filter] {aligned_dir.name}: kept={report['kept']} rejected={report['rejected']} "
        f"(drop={report['drop_ratio_effective']:.2%})"
    )
    return report


def run_extract(video: Path, frames_dir: Path, cfg: AutoTrainConfig, logger: TeeLogger) -> None:
    logger.log(f"[extract-video] {video.name} -> {frames_dir}")
    VideoEd.extract_video(str(video), str(frames_dir), output_ext="png", fps=cfg.extract_fps)


def run_align(frames_dir: Path, aligned_dir: Path, cfg: AutoTrainConfig, logger: TeeLogger) -> None:
    s3fd_path = Path(__file__).resolve().parent / "facelib" / "S3FD.npy"
    if s3fd_path.exists():
        logger.log(f"[extract-faces] S3FD weights path={s3fd_path} size_bytes={s3fd_path.stat().st_size}")
    else:
        logger.log(f"[extract-faces] WARNING: S3FD.npy missing at {s3fd_path}")
    logger.log(f"[extract-faces] {frames_dir.name} -> {aligned_dir}")
    Extractor.main(
        detector=cfg.detector,
        input_path=frames_dir,
        output_path=aligned_dir,
        output_debug=False,
        manual_fix=False,
        manual_output_debug_fix=False,
        manual_window_size=1368,
        face_type=cfg.face_type,
        max_faces_from_image=cfg.max_faces_from_image,
        image_size=cfg.extract_image_size,
        jpeg_quality=cfg.extract_jpeg_quality,
        cpu_only=cfg.cpu_only,
        force_gpu_idxs=cfg.force_gpu_idxs if not cfg.cpu_only else None,
    )


def run_xseg(aligned_dir: Path, cfg: AutoTrainConfig, logger: TeeLogger) -> None:
    if not cfg.enable_xseg:
        logger.log("[xseg] disabled by config")
        return
    if not cfg.xseg_model_dir.exists():
        logger.log(f"[xseg] model dir not found: {cfg.xseg_model_dir}, skip")
        return
    logger.log(f"[xseg] applying model to {aligned_dir}")
    orig_ask = nn.DeviceConfig.ask_choose_device
    try:
        nn.DeviceConfig.ask_choose_device = staticmethod(lambda choose_only_one=True: nn.DeviceConfig.BestGPU())
        with patched_io_defaults(logger):
            XSegUtil.apply_xseg(aligned_dir, cfg.xseg_model_dir)
    finally:
        nn.DeviceConfig.ask_choose_device = orig_ask


def run_faceset_enhance(aligned_dir: Path, cfg: AutoTrainConfig, logger: TeeLogger) -> None:
    if not cfg.enable_faceset_enhance:
        logger.log("[enhance] disabled by config")
        return
    logger.log(f"[enhance] {aligned_dir}")
    FacesetEnhancer.process_folder(
        aligned_dir,
        cpu_only=cfg.cpu_only,
        force_gpu_idxs=cfg.force_gpu_idxs if not cfg.cpu_only else None,
    )


def train_model(cfg: AutoTrainConfig, model_dir: Path, src_aligned: Path, dst_aligned: Path, logger: TeeLogger) -> Dict:
    logger.log("[train] initializing model")
    with patched_io_defaults(logger):
        model = models.import_model(cfg.model_name)(
            is_training=True,
            saved_models_path=model_dir,
            training_data_src_path=src_aligned,
            training_data_dst_path=dst_aligned,
            no_preview=True,
            force_model_name=cfg.model_alias,
            force_gpu_idxs=cfg.force_gpu_idxs if not cfg.cpu_only else None,
            cpu_only=cfg.cpu_only,
            silent_start=True,
            debug=False,
        )

    start = time.time()
    last_save = start
    last_backup = start
    best_loss = float("inf")
    best_iter = 0
    best_time = start
    report = {"stopped_reason": "", "best_loss": None, "best_iter": 0, "iters_done": 0, "quality_gate": {}}

    try:
        while True:
            current_iter, iter_time = model.train_one_iter()
            losses = model.get_loss_history()
            tail = losses[-cfg.loss_window_iters :] if losses else []
            flat_tail = [float(np.mean(x)) for x in tail] if tail else [9999.0]
            loss_avg = float(statistics.mean(flat_tail))

            if current_iter % 50 == 0:
                logger.log(
                    f"[train] iter={current_iter} iter_time={iter_time:.4f}s "
                    f"loss_avg(window={cfg.loss_window_iters})={loss_avg:.6f}"
                )

            if (best_loss - loss_avg) > cfg.min_improve_delta:
                best_loss = loss_avg
                best_iter = current_iter
                best_time = time.time()
                logger.log(f"[train] new best: iter={best_iter} loss={best_loss:.6f}")
                model.save()

            now = time.time()
            if now - last_save >= cfg.save_minutes * 60:
                logger.log("[train] periodic save")
                model.save()
                last_save = now

            if now - last_backup >= cfg.backup_minutes * 60:
                logger.log("[train] periodic backup")
                model.create_backup()
                last_backup = now

            elapsed_h = (now - start) / 3600.0
            no_improve_h = (now - best_time) / 3600.0
            if elapsed_h >= cfg.train_max_hours and current_iter >= cfg.min_iterations:
                report["stopped_reason"] = f"max_hours_reached({cfg.train_max_hours})"
                break
            if no_improve_h >= cfg.plateau_hours and current_iter >= cfg.min_iterations:
                report["stopped_reason"] = f"plateau_hours_reached({cfg.plateau_hours})"
                break
    finally:
        logger.log("[train] final save + finalize")
        model.save()
        model.finalize()

    report["best_loss"] = best_loss if best_loss < 1e9 else None
    report["best_iter"] = best_iter
    report["iters_done"] = model.get_iter()
    return report


def write_quality_reports(run_dir: Path, src_report: Dict, dst_report: Dict, train_report: Dict, logger: TeeLogger) -> Dict:
    quality = {
        "min_src_faces_required": int(os.environ.get("AUTOTRAIN_MIN_SRC_FACES", "300")),
        "min_dst_faces_required": int(os.environ.get("AUTOTRAIN_MIN_DST_FACES", "300")),
        "min_iterations_required": int(os.environ.get("AUTOTRAIN_MIN_ITERS", "2000")),
        "src_faces_kept": int(src_report.get("kept", 0)),
        "dst_faces_kept": int(dst_report.get("kept", 0)),
        "iterations_done": int(train_report.get("iters_done", 0)),
        "best_loss": train_report.get("best_loss"),
        "checks": {},
    }
    quality["checks"]["src_faces_ok"] = quality["src_faces_kept"] >= quality["min_src_faces_required"]
    quality["checks"]["dst_faces_ok"] = quality["dst_faces_kept"] >= quality["min_dst_faces_required"]
    quality["checks"]["iterations_ok"] = quality["iterations_done"] >= quality["min_iterations_required"]
    quality["checks"]["ready_for_export"] = all(quality["checks"].values())
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger.dump_json(reports_dir / "quality_report.json", quality)

    md = [
        "# Quality Report",
        "",
        f"- src_faces_kept: `{quality['src_faces_kept']}` / required `{quality['min_src_faces_required']}`",
        f"- dst_faces_kept: `{quality['dst_faces_kept']}` / required `{quality['min_dst_faces_required']}`",
        f"- iterations_done: `{quality['iterations_done']}` / required `{quality['min_iterations_required']}`",
        f"- best_loss: `{quality['best_loss']}`",
        f"- ready_for_export: `{quality['checks']['ready_for_export']}`",
    ]
    (reports_dir / "quality_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return quality


def export_dfm(cfg: AutoTrainConfig, model_dir: Path, output_dir: Path, logger: TeeLogger) -> Path:
    logger.log("[exportdfm] exporting")
    with patched_io_defaults(logger):
        ExportDFM.main(model_class_name=cfg.model_name, saved_models_path=model_dir)
    candidates = sorted(model_dir.glob("*.dfm"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("No .dfm file generated after export.")
    best = candidates[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / best.name
    shutil.copy2(best, final)
    logger.log(f"[exportdfm] done -> {final}")
    return final


def preset_defaults(name: str) -> Tuple[int, int]:
    # (extract image size, max faces from image)
    if name == "fast":
        return 384, 1
    if name == "quality":
        return 640, 1
    return 512, 1


def build_config(args: argparse.Namespace) -> AutoTrainConfig:
    image_size_default, max_faces_default = preset_defaults(args.preset)
    xseg_default = Path(r"E:\app\DeepFaceLab\DeepFaceLab_NVIDIA_RTX3000_series\_internal\model_generic_xseg")
    force_gpu = env_str("AUTOTRAIN_FORCE_GPU_IDXS", "0").strip()
    gpu_idxs = [int(x.strip()) for x in force_gpu.split(",") if x.strip()]
    model_name_raw = env_str("AUTOTRAIN_MODEL", "Model_SAEHD")
    model_name = model_name_raw[len("Model_") :] if model_name_raw.startswith("Model_") else model_name_raw

    return AutoTrainConfig(
        src_video=Path(args.videos[0]),
        dst_video=Path(args.videos[1]),
        xseg_model_dir=Path(env_str("AUTOTRAIN_XSEG_MODEL_DIR", str(xseg_default))),
        workdir=Path(env_str("AUTOTRAIN_WORKDIR", str(Path.cwd() / "autotrain_runs"))),
        model_name=model_name,
        model_alias=env_str("AUTOTRAIN_MODEL_ALIAS", "autotrain_saehd"),
        face_type=env_str("AUTOTRAIN_FACE_TYPE", "whole_face"),
        detector=env_str("AUTOTRAIN_DETECTOR", "s3fd"),
        extract_fps=env_int("AUTOTRAIN_FPS", 0),
        extract_image_size=env_int("AUTOTRAIN_EXTRACT_IMAGE_SIZE", image_size_default),
        extract_jpeg_quality=env_int("AUTOTRAIN_JPEG_QUALITY", 90),
        max_faces_from_image=env_int("AUTOTRAIN_MAX_FACES", max_faces_default),
        force_gpu_idxs=gpu_idxs,
        cpu_only=env_int("AUTOTRAIN_CPU_ONLY", 0) == 1,
        drop_ratio=max(0.0, min(0.9, env_float("AUTOTRAIN_DROP_RATIO", 0.20))),
        min_keep_ratio=max(0.1, min(0.95, env_float("AUTOTRAIN_MIN_KEEP_RATIO", 0.50))),
        train_max_hours=max(0.1, env_float("AUTOTRAIN_MAX_HOURS", 1.0)),
        plateau_hours=max(0.1, env_float("AUTOTRAIN_PLATEAU_HOURS", 1.0)),
        backup_minutes=max(1, env_int("AUTOTRAIN_BACKUP_MIN", 10)),
        save_minutes=max(1, env_int("AUTOTRAIN_SAVE_MIN", 10)),
        loss_window_iters=max(20, env_int("AUTOTRAIN_LOSS_WINDOW", 200)),
        min_improve_delta=max(0.0, env_float("AUTOTRAIN_MIN_IMPROVE_DELTA", 0.0001)),
        best_criterion=env_str("AUTOTRAIN_BEST_CRITERION", "loss_lowest"),
        preset=args.preset,
        enable_xseg=env_int("AUTOTRAIN_ENABLE_XSEG", 1) == 1,
        enable_faceset_enhance=env_int("AUTOTRAIN_ENABLE_ENHANCE", 1) == 1,
        preprocess_parallel=env_int("AUTOTRAIN_PREPROCESS_PARALLEL", 1) == 1,
        parallel_align=env_int("AUTOTRAIN_PARALLEL_ALIGN", 0) == 1,
        parallel_enhance=env_int("AUTOTRAIN_PARALLEL_ENHANCE", 0) == 1,
        train_batch_size=max(1, env_int("AUTOTRAIN_BATCH_SIZE", 8)),
        min_iterations=max(100, env_int("AUTOTRAIN_MIN_ITERS", 2000)),
        min_src_faces=max(10, env_int("AUTOTRAIN_MIN_SRC_FACES", 300)),
        min_dst_faces=max(10, env_int("AUTOTRAIN_MIN_DST_FACES", 300)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepFaceLab end-to-end auto train without subprocess main.py calls.")
    parser.add_argument(
        "videos",
        nargs=2,
        help="Input videos in order: <src_video> <dst_video>. Example: 1.MOV 2.mp4",
    )
    parser.add_argument("--preset", choices=["fast", "balanced", "quality"], default="balanced")
    return parser.parse_args()


def run_with_config(cfg: AutoTrainConfig) -> Dict:
    root = Path.cwd()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.workdir / run_id
    src_frames = run_dir / "data_src"
    dst_frames = run_dir / "data_dst"
    src_aligned = src_frames / "aligned"
    dst_aligned = dst_frames / "aligned"
    model_dir = run_dir / "model"
    out_dir = run_dir / "outputs"
    report_dir = run_dir / "reports"
    for p in [src_frames, dst_frames, src_aligned, dst_aligned, model_dir, out_dir, report_dir]:
        p.mkdir(parents=True, exist_ok=True)

    logger = TeeLogger(run_dir / "autotrain.log")
    logger.log("=== AUTOTRAIN START ===")
    logger.log(f"root={root}")
    logger.log(f"run_dir={run_dir}")
    logger.log(f"src={cfg.src_video} dst={cfg.dst_video}")
    logger.log(f"preset={cfg.preset} model={cfg.model_name} best_criterion={cfg.best_criterion}")

    src_video = cfg.src_video if cfg.src_video.is_absolute() else (root / cfg.src_video)
    dst_video = cfg.dst_video if cfg.dst_video.is_absolute() else (root / cfg.dst_video)
    if not src_video.exists() or not dst_video.exists():
        logger.log(f"Input video missing. src_exists={src_video.exists()} dst_exists={dst_video.exists()}")
        return 2
    cfg.src_video, cfg.dst_video = src_video, dst_video

    nn.initialize_main_env()

    try:
        if cfg.preprocess_parallel:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = [
                    ex.submit(run_extract, cfg.src_video, src_frames, cfg, logger),
                    ex.submit(run_extract, cfg.dst_video, dst_frames, cfg, logger),
                ]
                for fut in futures:
                    fut.result()
        else:
            run_extract(cfg.src_video, src_frames, cfg, logger)
            run_extract(cfg.dst_video, dst_frames, cfg, logger)

        if cfg.parallel_align:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = [
                    ex.submit(run_align, src_frames, src_aligned, cfg, logger),
                    ex.submit(run_align, dst_frames, dst_aligned, cfg, logger),
                ]
                for fut in futures:
                    fut.result()
        else:
            run_align(src_frames, src_aligned, cfg, logger)
            run_align(dst_frames, dst_aligned, cfg, logger)

        src_report = filter_faceset(src_aligned, cfg, logger, report_dir / "src_filter_report.json")
        dst_report = filter_faceset(dst_aligned, cfg, logger, report_dir / "dst_filter_report.json")
        logger.log(f"[filter-summary] src={src_report} dst={dst_report}")
        if src_report.get("kept", 0) < cfg.min_src_faces or dst_report.get("kept", 0) < cfg.min_dst_faces:
            raise RuntimeError(
                f"Not enough aligned faces after filtering. "
                f"src={src_report.get('kept', 0)} (min={cfg.min_src_faces}), "
                f"dst={dst_report.get('kept', 0)} (min={cfg.min_dst_faces})"
            )

        if cfg.parallel_enhance:
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = [
                    ex.submit(run_faceset_enhance, src_aligned, cfg, logger),
                    ex.submit(run_faceset_enhance, dst_aligned, cfg, logger),
                ]
                for fut in futures:
                    fut.result()
        else:
            run_faceset_enhance(src_aligned, cfg, logger)
            run_faceset_enhance(dst_aligned, cfg, logger)

        run_xseg(src_aligned, cfg, logger)
        run_xseg(dst_aligned, cfg, logger)

        train_report = train_model(cfg, model_dir, src_aligned, dst_aligned, logger)
        logger.dump_json(report_dir / "train_report.json", train_report)
        logger.log(f"[train-report] {train_report}")
        quality = write_quality_reports(run_dir, src_report, dst_report, train_report, logger)
        if not quality["checks"]["ready_for_export"]:
            raise RuntimeError(f"Quality gate failed: {quality}")

        dfm_path = export_dfm(cfg, model_dir, out_dir, logger)
        summary = {
            "run_dir": str(run_dir),
            "dfm_output": str(dfm_path),
            "log_file": str(run_dir / "autotrain.log"),
            "reports": str(report_dir),
            "model_dir": str(model_dir),
            "quality_report": str(report_dir / "quality_report.json"),
        }
        logger.dump_json(run_dir / "summary.json", summary)
        logger.dump_json(run_dir / "metrics.json", {"train_report": train_report, "quality": quality})
        summary_md = [
            "# Run Summary",
            "",
            f"- run_dir: `{run_dir}`",
            f"- dfm_output: `{dfm_path}`",
            f"- iterations_done: `{train_report.get('iters_done')}`",
            f"- best_loss: `{train_report.get('best_loss')}`",
            f"- ready_for_export: `{quality['checks']['ready_for_export']}`",
        ]
        (run_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")
        timeline = [
            "# Job Timeline",
            "",
            f"- `{datetime.now().isoformat()}Z` Extract/align/filter completed",
            f"- `{datetime.now().isoformat()}Z` Training completed at iter `{train_report.get('iters_done')}`",
            f"- `{datetime.now().isoformat()}Z` Quality gate passed and DFM exported: `{dfm_path.name}`",
        ]
        (run_dir / "timeline.md").write_text("\n".join(timeline) + "\n", encoding="utf-8")
        logger.log(f"[done] summary={summary}")
        logger.log("=== AUTOTRAIN END (SUCCESS) ===")
        return {"ok": True, "run_dir": str(run_dir), "summary": summary}
    except Exception as e:
        logger.log(f"[fatal] {type(e).__name__}: {e}")
        logger.log("=== AUTOTRAIN END (FAILED) ===")
        raise


def run_pipeline(videos: List[str], preset: str = "balanced", env_overrides: Dict[str, str] | None = None) -> Dict:
    old_env = {}
    if env_overrides:
        for k, v in env_overrides.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = str(v)
    try:
        args = argparse.Namespace(videos=videos, preset=preset)
        cfg = build_config(args)
        return run_with_config(cfg)
    finally:
        if env_overrides:
            for k in env_overrides.keys():
                prev = old_env.get(k)
                if prev is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = prev


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    run_with_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
