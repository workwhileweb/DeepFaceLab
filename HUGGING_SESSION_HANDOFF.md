# DeepFaceLab on Hugging Face - Session Handoff

## Muc tieu da hoan thanh

- Dong goi DeepFaceLab thanh API service tren Hugging Face Spaces (Docker + FastAPI).
- Chay pipeline train tu dong tu upload video den export `.dfm`.
- Tach worker train ra subprocess de API khong bi block/crash.
- Toi uu toc do train (parallel preprocess + batch size control).
- Xu ly cac loi deployment/runtime tren Hugging Face (import, dependency, storage I/O, restart).
- Chay thanh cong job tren GPU `l4x1` va tao link tai `.dfm`.

## Kien truc hien tai

- `runonhugging.py`
  - FastAPI API (`/healthz`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/logs`, `/jobs/{id}/download`).
  - Khi tao job, API spawn `runonhugging_worker.py` bang `subprocess.Popen`.
  - Co check PID worker trong API status; neu worker chet bat thuong -> mark job `failed`.
  - Data root dang duoc cau hinh de uu tien ep vao env (`DFL_DATA_ROOT`), hien tai set `/tmp` trong Space.

- `runonhugging_worker.py`
  - Nhan args job, set env overrides va goi `autotrain.run_pipeline(...)`.
  - Cap nhat status file theo tien trinh (`queued` -> `running` -> `completed`/`failed`).
  - Truyen cac bien toi uu:
    - `AUTOTRAIN_PREPROCESS_PARALLEL`
    - `AUTOTRAIN_PARALLEL_ALIGN`
    - `AUTOTRAIN_PARALLEL_ENHANCE`
    - `AUTOTRAIN_BATCH_SIZE` (default da de o muc cao hon de tan dung GPU).

- `autotrain.py`
  - Bo sung `ThreadPoolExecutor` de chay song song cac buoc preprocess (extract/align/enhance).
  - Bo sung config doc tu env cho che do parallel.
  - Auto-input batch size theo `AUTOTRAIN_BATCH_SIZE`.

- `hugging_space/Dockerfile`
  - Chuyen code runtime sang `/workspace` (khong dat code trong `/app` vi `/app` la bucket mount).
  - Set bien moi truong de tranh loi `numexpr`:
    - `OMP_NUM_THREADS=1`
    - `OPENBLAS_NUM_THREADS=1`
    - `NUMEXPR_NUM_THREADS=1`
  - Set `DFL_DATA_ROOT=/tmp` de workload I/O nang khong ghi truc tiep len bucket mount.

## Cac loi da gap va cach fix

1. `ModuleNotFoundError: localization`
   - Upload them module/folder thieu (`localization`, `merger`) len Space.

2. `ValueError: invalid literal for int() with base 10: '3500m'` (numexpr/OpenMP)
   - Hard-set thread env vars ve so nguyen an toan trong Docker.

3. `ERROR: Could not import module "runonhugging"`
   - Nguyen nhan: code bi mount `/app` che mat.
   - Fix: copy/run code o `/workspace`, de `/app` cho volume data.

4. `OSError: [Errno 5] Input/output error` khi train/finalize model
   - Nguyen nhan: ghi I/O nang vao bucket mount `/app`.
   - Fix: cho pipeline ghi du lieu train vao `/tmp` (`DFL_DATA_ROOT=/tmp`).

5. Job bi mat (`404 job not found`) trong luc chuyen hardware
   - Nguyen nhan: Space restart khi migrate GPU tier.
   - Cach van hanh: chi submit job khi runtime `RUNNING` va `current == requested` (vi du `l4x1`).

## Ket qua lan chay gan nhat

- Space: `cristiefisherwfc43821s/deepfacelab-autotrain-api`
- Runtime da on dinh tren: `l4x1`
- SHA deploy app: `e39c662de438c9bda1e1f5babde4608d8b7079a2`
- Job thanh cong:
  - `job_id`: `12f8edf0f6fb`
  - `status`: `completed`
  - Download: `https://cristiefisherwfc43821s-deepfacelab-autotrain-api.hf.space/jobs/12f8edf0f6fb/download`

## Luu y quan trong cho session tiep theo

- Artifact train dang nam trong `/tmp` (ephemeral), can tai ngay sau khi xong.
- Neu can giu ket qua lau dai:
  - Download `.dfm` ve local, hoac
  - Copy artifact sang storage ben ngoai (bucket/object storage) sau khi train xong.
- Tranh submit job khi Space dang `RUNNING_APP_STARTING` hoac dang migrate hardware.
- Theo doi `/healthz` truoc khi goi `POST /jobs`.

## Quy trinh van hanh de tiep tuc

1. Xac nhan runtime:
   - Stage: `RUNNING`
   - Hardware `current` trung `requested`
2. Kiem tra health:
   - `GET /healthz`
3. Submit job:
   - `POST /jobs` voi `src_video`, `dst_video`, `preset`, `max_hours`, `plateau_hours`
4. Poll:
   - `GET /jobs/{id}`
   - `GET /jobs/{id}/logs?tail_lines=...`
5. Khi `completed`:
   - Lay link `GET /jobs/{id}/download`
   - Tai file ngay.

## Files da tac dong trong session

- `runonhugging.py`
- `runonhugging_worker.py`
- `autotrain.py`
- `hugging_space/Dockerfile`

