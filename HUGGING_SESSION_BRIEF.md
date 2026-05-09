# Hugging Session Brief (10 lines)

1. Muc tieu: chay DeepFaceLab autotrain qua API tren Hugging Face Spaces va xuat `.dfm`.
2. API chinh o `runonhugging.py`; worker train tach rieng o `runonhugging_worker.py` (spawn subprocess).
3. Da fix crash `numexpr` trong Docker bang: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.
4. Da fix loi import app tren HF bang cach chay source o `/workspace` (khong dat source code trong `/app`).
5. Da fix `OSError: [Errno 5]` bang `DFL_DATA_ROOT=/tmp` de train ghi vao local ephemeral storage.
6. `autotrain.py` da toi uu preprocess song song (extract/align/enhance) + batch size qua `AUTOTRAIN_BATCH_SIZE`.
7. Luu y: `/tmp` la ephemeral, can tai `.dfm` ngay sau khi job `completed`.
8. Chi submit job khi Space on dinh: `stage=RUNNING` va hardware `current==requested` (vi du `l4x1`).
9. Job thanh cong gan nhat: `12f8edf0f6fb` (status `completed`).
10. Link tai truc tiep: `https://cristiefisherwfc43821s-deepfacelab-autotrain-api.hf.space/jobs/12f8edf0f6fb/download`.

