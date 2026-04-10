# Huong Dan Train Face Model Bang DeepFaceLab

Tai lieu nay huong dan quy trinh train model face swap voi DeepFaceLab theo 2 cach:
- Dau vao la **video**
- Dau vao la **thu muc anh**

Noi dung duoi day dung theo CLI trong repo nay (`main.py`).

## 1) Chuan bi moi truong

- Python: `>=3.6` (nen dung 3.8/3.10)
- GPU NVIDIA + CUDA (khuyen nghi), hoac co the chay CPU (rat cham)
- Tao virtual environment va cai dependency theo huong dan ban dang su dung trong may
- Chuan bi du lieu:
  - `SRC`: khuon mat muon dua vao
  - `DST`: video/anh dich muon thay khuon mat

Goi y cau truc thu muc:

```text
workspace/
  data_src/              # anh frame SRC
  data_dst/              # anh frame DST
  data_src/aligned/      # face da extract SRC
  data_dst/aligned/      # face da extract DST
  model/                 # checkpoint model
  merged/                # ket qua frame sau merge
  merged_mask/           # mask frame
```

## 2) Truong hop A - Train tu VIDEO

### Buoc A1. Cat frame tu video

Cat video SRC:

```bash
python main.py videoed extract-video --input-file "E:/data/src.mp4" --output-dir "E:/work/data_src" --output-ext png --fps 0
```

Cat video DST:

```bash
python main.py videoed extract-video --input-file "E:/data/dst.mp4" --output-dir "E:/work/data_dst" --output-ext png --fps 0
```

`--fps 0` = lay full fps. Neu may yeu hoac video dai, co the dat `--fps 5` hoac `--fps 10`.

### Buoc A2. Extract face tu frame

Extract SRC:

```bash
python main.py extract --detector s3fd --input-dir "E:/work/data_src" --output-dir "E:/work/data_src/aligned" --face-type whole_face
```

Extract DST:

```bash
python main.py extract --detector s3fd --input-dir "E:/work/data_dst" --output-dir "E:/work/data_dst/aligned" --face-type whole_face
```

Neu bi miss mat, dung manual fix:

```bash
python main.py extract --detector manual --manual-fix --input-dir "E:/work/data_dst" --output-dir "E:/work/data_dst/aligned"
```

### Buoc A3. Loc faceset (khuyen nghi)

Sap xep bo anh theo do mo:

```bash
python main.py sort --input-dir "E:/work/data_src/aligned" --by final-by-blur
python main.py sort --input-dir "E:/work/data_dst/aligned" --by final-by-blur
```

Sau do xoa cac anh blur, che mat, sai goc qua muc.

### Buoc A4. Train model

Vi du train voi model SAEHD:

```bash
python main.py train --training-data-src-dir "E:/work/data_src/aligned" --training-data-dst-dir "E:/work/data_dst/aligned" --model-dir "E:/work/model" --model Model_SAEHD
```

Neu chay CPU:

```bash
python main.py train --training-data-src-dir "E:/work/data_src/aligned" --training-data-dst-dir "E:/work/data_dst/aligned" --model-dir "E:/work/model" --model Model_SAEHD --cpu-only
```

Train den khi preview on dinh (thuong vai chuc nghin den tram nghin iteration tuy du lieu/GPU).

### Buoc A5. Merge vao frame DST

```bash
python main.py merge --input-dir "E:/work/data_dst" --output-dir "E:/work/merged" --output-mask-dir "E:/work/merged_mask" --aligned-dir "E:/work/data_dst/aligned" --model-dir "E:/work/model" --model Model_SAEHD
```

### Buoc A6. Dung frame da merge de tao lai video

```bash
python main.py videoed video-from-sequence --input-dir "E:/work/merged" --output-file "E:/work/result.mp4" --reference-file "E:/data/dst.mp4" --ext png --include-audio
```

## 3) Truong hop B - Train tu THU MUC ANH

Neu ban da co san thu muc anh (khong can buoc tach frame tu video):

1. Dat anh SRC vao `data_src/`, anh DST vao `data_dst/`
2. Chay tu Buoc A2 den A5 nhu tren
3. Neu muc tieu chi la sinh anh (khong can video), ban co the dung ngay output trong `merged/`

Lenh toi thieu:

```bash
python main.py extract --detector s3fd --input-dir "E:/work/data_src" --output-dir "E:/work/data_src/aligned" --face-type whole_face
python main.py extract --detector s3fd --input-dir "E:/work/data_dst" --output-dir "E:/work/data_dst/aligned" --face-type whole_face
python main.py train --training-data-src-dir "E:/work/data_src/aligned" --training-data-dst-dir "E:/work/data_dst/aligned" --model-dir "E:/work/model" --model Model_SAEHD
python main.py merge --input-dir "E:/work/data_dst" --output-dir "E:/work/merged" --output-mask-dir "E:/work/merged_mask" --aligned-dir "E:/work/data_dst/aligned" --model-dir "E:/work/model" --model Model_SAEHD
```

## 4) Meo quan trong de ket qua dep

- So luong anh `aligned` nen > 2,000 moi ben (cang nhieu cang tot)
- Anh can da dang goc quay, anh sang, bieu cam
- Uu tien nguon video/anh ro net, it motion blur
- Khong nen train qua it du lieu (de bi mo, rung, sai expression)
- Co the dung `xseg` de cai thien mask vi nhanh toc:
  - `python main.py xseg editor --input-dir ".../aligned"`
  - `python main.py xseg apply --input-dir ".../aligned" --model-dir ".../model"`

## 5) Loi thuong gap

- **CUDA out of memory**: giam batch size/resolution trong cau hinh model, hoac dung GPU manh hon
- **Extract sai mat / nham mat**: dung `--manual-fix`, loc lai faceset, xoa anh loi
- **Merge bi vien mat xau**: train them, dung XSeg, chinh tham so merge
- **Train qua cham**: kiem tra co dang roi vao CPU mode khong

## 6) Mau lenh tong hop nhanh

```bash
# 1) video -> frames
python main.py videoed extract-video --input-file "E:/data/src.mp4" --output-dir "E:/work/data_src" --output-ext png --fps 0
python main.py videoed extract-video --input-file "E:/data/dst.mp4" --output-dir "E:/work/data_dst" --output-ext png --fps 0

# 2) extract faces
python main.py extract --detector s3fd --input-dir "E:/work/data_src" --output-dir "E:/work/data_src/aligned" --face-type whole_face
python main.py extract --detector s3fd --input-dir "E:/work/data_dst" --output-dir "E:/work/data_dst/aligned" --face-type whole_face

# 3) train
python main.py train --training-data-src-dir "E:/work/data_src/aligned" --training-data-dst-dir "E:/work/data_dst/aligned" --model-dir "E:/work/model" --model Model_SAEHD

# 4) merge
python main.py merge --input-dir "E:/work/data_dst" --output-dir "E:/work/merged" --output-mask-dir "E:/work/merged_mask" --aligned-dir "E:/work/data_dst/aligned" --model-dir "E:/work/model" --model Model_SAEHD

# 5) image sequence -> video
python main.py videoed video-from-sequence --input-dir "E:/work/merged" --output-file "E:/work/result.mp4" --reference-file "E:/data/dst.mp4" --ext png --include-audio
```

---

Neu ban muon, toi co the viet them mot file huong dan rieng cho:
- quy trinh cho may yeu (VRAM thap),
- preset tham so SAEHD cho nguoi moi,
- checklist chat luong truoc khi merge.
