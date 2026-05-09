# Hugging Face Docker Space Deploy

## Runtime requirements

- GPU Space (recommended `l4x1` or higher)
- Persistent storage: many Spaces mount the Hub bucket at **`/app`**. This image keeps **code in `/workspace`** and sets **`DFL_DATA_ROOT=/app`** so training data and jobs persist without hiding the app.
- Docker SDK Space

## Required environment variables

- `DFL_DATA_ROOT=/app` (or your bucket mount path; must match Space volume `mount_path`)
- `HF_EXIT_WHEN_DONE=1`
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `AUTOTRAIN_MIN_ITERS=2000`
- `AUTOTRAIN_MIN_SRC_FACES=300`
- `AUTOTRAIN_MIN_DST_FACES=300`

## Docker build

The Space `Dockerfile` installs from `requirements-hf-space.txt` (FastAPI + TensorFlow 2.15 + DFL deps). TensorFlow is pinned to 2.15.x so `typing-extensions` stays compatible with FastAPI (TF 2.12 required `typing-extensions<4.6`, which conflicts with current FastAPI). That file must exist in the Space repo root or the build fails at `COPY`.

## Deploy flow

1. Push this repository with `Dockerfile`, `requirements-hf-space.txt`, `runonhugging.py`, `runonhugging_worker.py`, and `autotrain.py`.
2. Ensure Space has GPU hardware and a persistent volume (commonly mounted at `/app`).
3. Wait for runtime `RUNNING`, then check `GET /healthz`.
4. Submit job with `POST /jobs` (`src_video`, `dst_video`, `preset`, `max_hours`, `plateau_hours`).
5. Monitor with `GET /jobs/{id}` and `GET /jobs/{id}/logs`.
6. Download result with `GET /jobs/{id}/download` or `GET /jobs/{id}/bundle`.

## Data layout

With `DFL_DATA_ROOT=/app` (default in `Dockerfile`):

- `/app/hf_service/jobs/<job_id>/inputs`
- `/app/hf_service/jobs/<job_id>/runs/<run_id>`
- `/app/hf_service/jobs/<job_id>/status.json`
- `/app/hf_service/jobs/<job_id>/timeline.md`

Application code lives under `/workspace` in the container image.
