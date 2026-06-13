# 🃏 UNO Card Classifier

Sistem klasifikasi kartu UNO menggunakan **Transfer Learning MobileNetV2** (TensorFlow/Keras) dengan antarmuka web **Flask** — mendukung upload gambar dan deteksi real-time via webcam.

> **Test Accuracy: 97.88%** · 54 kelas · CPU-only compatible

---

## ⚙️ Instalasi

```bash
# Python 3.9 — gunakan virtual environment
source /root/kholid/python_env/bin/activate

# Install dependensi dengan versi yang sudah diuji
pip install "tensorflow==2.12.0" "keras==2.12.0"
pip install "numpy==1.23.5" "protobuf==3.20.3" "h5py==3.9.0"
pip install "urllib3<2.0"   # wajib untuk OpenSSL 1.0.2
pip install flask Pillow scikit-learn matplotlib seaborn pyyaml
```

> **Catatan sistem:** GCC 4.8.5 + Python 3.9 + OpenSSL 1.0.2 → wajib pin versi di atas.
> TensorFlow ≥ 2.13 tidak kompatibel karena membutuhkan C++17 dan OpenSSL 1.1.1+.

---

## 🗂️ Persiapan Dataset

Dataset yang digunakan (format YOLO, dari Roboflow):

| Dataset | Kelas | Split |
|---|---|---|
| `New Dataset` (54 kelas per warna) | lblue-1, lred-skip, lcolor-40, … | train/valid/test |
| `UNO-DATASET--3` (54 kelas) | lblue-0, lred-20, lcolor-400, … | train/valid/test |
| `UNO-DATASET--2` (54 kelas, 3 bg) | plain_bg, dark_bg, non-plain_bg | images/labels |

### Konversi YOLO → Keras

```bash
python yolo_to_keras.py
```

Script ini akan:
1. Membaca label YOLO (`.txt` berisi `class_id cx cy w h`)
2. Crop setiap bounding box dari gambar asli
3. **Resize crop ke 224×224 dengan padding hitam** (konsisten dengan input MobileNetV2)
4. Normalisasi nama kelas: `lblue-20` → `blue_draw_two`, `lcolor-400` → `wild_draw_four`, dst.
5. Menyimpan ke `keras_dataset/<nama_kelas>/`


---

## 🚀 Training

```bash
python train.py
```

### Alur Pipeline

```
keras_dataset/
    ↓ ImageDataGenerator (augmentasi + split 80/10/10)
    ↓ MobileNetV2 pretrained ImageNet
    ↓ Fase 1: Feature Extraction (base frozen, head dilatih)
    ↓ Fase 2: Fine-Tuning (30 layer terakhir di-unfreeze, LR kecil)
    ↓ Evaluasi + confusion matrix + training history plot
    ↓ Simpan uno_mobilenetv2_final.h5
```

### Arsitektur Model

```
Input (224×224×3)
    ↓
MobileNetV2 — pretrained ImageNet (2.26M params, non-trainable di fase 1)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU) + L2(0.001)
    ↓
Dropout(0.4)
    ↓
Dense(num_classes, Softmax)
```

| Fase | Yang Dilatih | Learning Rate | Epoch | Early Stop |
|------|---|---|---|---|
| 1 — Feature Extraction | Head saja | 0.001 | 40 | patience=10 |
| 2 — Fine-Tuning | Head + 30 layer terakhir | 0.0001 | 40 | patience=10 |

### Hasil Training

| Metrik | Nilai |
|---|---|
| Test Accuracy | **97.88%** |
| Test Loss | 0.0980 |
| Jumlah kelas | 54 |

---

## 🌐 Menjalankan Web App

```bash
python app.py
# → http://localhost:5000
```
---

## 🌐 REST API

### `POST /predict`

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@kartu.jpg"
```

**Response:**
```json
{
  "class": "blue_7",
  "display": "Blue 7",
  "confidence": 94.8,
  "color": "#1E90FF",
  "top3": [
    { "class": "blue_7",   "display": "Blue 7",   "confidence": 94.8 },
    { "class": "blue_1",   "display": "Blue 1",   "confidence": 3.1  },
    { "class": "green_7",  "display": "Green 7",  "confidence": 1.4  }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "num_classes": 54 }
```

### `GET /classes`

```json
{ "classes": ["blue_0", "blue_1", ...], "total": 54 }
```

---

## 🎯 Kelas yang Didukung (54 kelas)

| Warna | Angka | Aksi |
|---|---|---|
| 🔴 Red | 0–9 | skip, reverse, draw_two |
| 🔵 Blue | 0–9 | skip, reverse, draw_two |
| 🟢 Green | 0–9 | skip, reverse, draw_two |
| 🟡 Yellow | 0–9 | skip, reverse, draw_two |
| 🟣 Wild | — | wild, wild_draw_four |

Format nama kelas: `<warna>_<angka_atau_aksi>` — contoh: `red_7`, `blue_skip`, `wild_draw_four`


---

## 📝 Catatan Pengembangan

- **Mengapa akurasi webcam lebih rendah?** Model dilatih dengan crop kartu dari dataset foto (kartu kecil di foto ramai). Foto langsung dari kamera memiliki distribusi berbeda. Solusi jangka panjang: tambahkan foto kamera sendiri ke dataset dan training ulang.
- **Mengapa tidak pakai TensorFlow terbaru?** Sistem menggunakan GCC 4.8.5 dan OpenSSL 1.0.2 yang tidak kompatibel dengan dependensi TF ≥ 2.13 (`ml-dtypes`, `optree` yang butuh C++17/C++20).
- **Format nama kelas:** Dataset asli pakai format `lblue-7`, `lredskip-20`, dll. Script `yolo_to_keras.py` menormalisasi ke format standar `blue_7`, `red_skip`, dst.
