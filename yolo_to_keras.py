import shutil
import yaml
from pathlib import Path
from PIL import Image

DATASETS = [
    {
        "name": "NewDataset",
        "yaml": "/root/kholid/uno/dataset/New Dataset/data.yaml",
        "splits": {
            "train": "/root/kholid/uno/dataset/New Dataset/train",
            "valid": "/root/kholid/uno/dataset/New Dataset/valid",
            "test":  "/root/kholid/uno/dataset/New Dataset/test",
        }
    },
    {
        "name": "Dataset3",
        "yaml": "/root/kholid/uno/dataset/UNO-DATASET--3/data.yaml",
        "splits": {
            "train": "/root/kholid/uno/dataset/UNO-DATASET--3/train",
            "valid": "/root/kholid/uno/dataset/UNO-DATASET--3/valid",
            "test":  "/root/kholid/uno/dataset/UNO-DATASET--3/test",
        }
    },
    {
        "name": "Dataset2",
        "yaml": "/root/kholid/uno/dataset/UNO-DATASET--2/dark_bg/data.yaml",
        "splits": {
            "plain":    "/root/kholid/uno/dataset/UNO-DATASET--2/plain_bg",
            "dark":     "/root/kholid/uno/dataset/UNO-DATASET--2/dark_bg",
            "nonplain": "/root/kholid/uno/dataset/UNO-DATASET--2/non-plain_bg",
        }
    },
]

OUTPUT_DIR = "/root/kholid/uno/keras_dataset"
MARGIN     = 10

def normalize_class_name(raw: str) -> str:
    s = raw.lower().strip()

    if s.startswith('lcolor'):
        # lcolor2-50 atau lcolor-400 → wild draw four
        # lcolor-40 → wild biasa
        remainder = s[len('lcolor'):]
        # Jika ada '2' setelah 'lcolor' atau angka >= 400 → wild draw four
        if remainder.startswith('2') or '400' in remainder or '50' in remainder:
            return 'wild_draw_four'
        return 'wild'

    color_prefixes = [
        ('lblue',   'blue'),
        ('lgreen',  'green'),
        ('lred',    'red'),
        ('lyellow', 'yellow'),
    ]

    for prefix, color in color_prefixes:
        if not s.startswith(prefix):
            continue

        remainder = s[len(prefix):]  # bagian setelah nama warna

        # Aksi: reverse
        if remainder.startswith('revers'):
            return f'{color}_reverse'

        # Aksi: skip
        if remainder.startswith('skip'):
            return f'{color}_skip'

        # Aksi: flip (New Dataset) atau angka 20 → draw two
        if remainder.startswith('flip'):
            return f'{color}_draw_two'

        # Angka setelah '-'
        # remainder bisa: '-1', '-20', '-0', '2-50' (jarang)
        num_str = remainder.lstrip('-').split('-')[0]
        if num_str.isdigit():
            num = int(num_str)
            if num == 20:
                return f'{color}_draw_two'
            if num == 10:
                num = 0   # beberapa dataset pakai 10 untuk kartu 0
            return f'{color}_{num}'

        break  # prefix cocok tapi format tidak dikenali

    # Fallback: bersihkan saja karakter aneh
    return raw.lower().replace('-', '_').replace(' ', '_')

def load_class_names(yaml_path: str) -> list:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    for key in ('names', 'nc_names', 'classes'):
        if key in data:
            names = data[key]
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names.keys())]
            return names
    raise ValueError(f"Key 'names' tidak ditemukan di {yaml_path}")


def yolo_box_to_pixel(box, img_w, img_h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2


def crop_and_save(img_path, label_path, class_names, output_dir,
                  margin=10, prefix=''):
    try:
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
    except Exception:
        return 0

    if not Path(label_path).exists():
        return 0

    with open(label_path, 'r') as f:
        lines = f.read().strip().splitlines()

    saved = 0
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        if class_id >= len(class_names):
            continue

        try:
            box = list(map(float, parts[1:5]))
            x1, y1, x2, y2 = yolo_box_to_pixel(box, img_w, img_h)
        except ValueError:
            continue

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_w, x2 + margin)
        y2 = min(img_h, y2 + margin)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img.crop((x1, y1, x2, y2))

        cls_name  = normalize_class_name(class_names[class_id])
        cls_dir   = Path(output_dir) / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{prefix}{Path(img_path).stem}_b{i}.jpg"
        TARGET = 224
        cw, ch = crop.size
        scale = TARGET / max(cw, ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        crop_resized = crop.resize((new_w, new_h), Image.LANCZOS)
        
        # Padding ke 224x224
        padded = Image.new('RGB', (TARGET, TARGET), (0, 0, 0))
        pad_x = (TARGET - new_w) // 2
        pad_y = (TARGET - new_h) // 2
        padded.paste(crop_resized, (pad_x, pad_y))
        padded.save(cls_dir / out_name, 'JPEG', quality=95)
        saved += 1

    return saved


def process_dataset(ds_cfg, output_dir, margin=10):
    name      = ds_cfg['name']
    yaml_path = ds_cfg['yaml']
    splits    = ds_cfg['splits']

    print(f"\n{'─'*60}")
    print(f"📦 {name}")
    print(f"{'─'*60}")

    if not Path(yaml_path).exists():
        print(f"  ❌ YAML tidak ditemukan: {yaml_path}")
        return 0

    class_names = load_class_names(yaml_path)

    # Preview normalisasi 5 kelas pertama
    print(f"  Contoh normalisasi nama kelas:")
    for raw in class_names[:5]:
        print(f"    {raw:<25} → {normalize_class_name(raw)}")

    total = 0
    for split_name, split_dir in splits.items():
        img_dir   = Path(split_dir) / 'images'
        label_dir = Path(split_dir) / 'labels'

        if not img_dir.exists():
            print(f"  ⚠️  [{split_name}] tidak ditemukan: {img_dir}")
            continue

        imgs = sorted(list(img_dir.glob('*.jpg')) +
                      list(img_dir.glob('*.jpeg')) +
                      list(img_dir.glob('*.png')))

        print(f"  [{split_name.upper():5}] {len(imgs):4d} gambar", end='', flush=True)

        n = 0
        for img_path in imgs:
            label_path = label_dir / (img_path.stem + '.txt')
            n += crop_and_save(img_path, label_path, class_names,
                               output_dir, margin, prefix=f'{name[:4]}_')
        total += n
        print(f"  →  {n} crop")

    print(f"  ✅ Total dari dataset ini: {total} crop")
    return total


def print_summary(output_dir):
    print(f"\n{'═'*60}")
    print(f"📊 DISTRIBUSI KELAS FINAL")
    print(f"{'═'*60}")
    print(f"   {'Kelas':<28} {'Gambar':>7}")
    print(f"   {'─'*28} {'─'*7}")

    class_dirs = sorted(Path(output_dir).iterdir())
    grand_total = 0
    low_count   = []

    for d in class_dirs:
        if not d.is_dir():
            continue
        count = len(list(d.glob('*.jpg')))
        grand_total += count
        flag = ' ⚠️' if count < 30 else ''
        print(f"   {d.name:<28} {count:>7}{flag}")
        if count < 30:
            low_count.append(d.name)

    print(f"   {'─'*28} {'─'*7}")
    print(f"   {'TOTAL':<28} {grand_total:>7}")
    print(f"\n   Jumlah kelas  : {len(class_dirs)}")
    print(f"   Total gambar  : {grand_total}")
    avg = grand_total // max(len(class_dirs), 1)
    print(f"   Rata-rata     : {avg} gambar/kelas")

    if low_count:
        print(f"\n  ⚠️  Kelas dengan <30 gambar ({len(low_count)} kelas):")
        for c in low_count:
            print(f"     - {c}")

    print(f"\n🚀 Jalankan training dengan:")
    print(f"   DATA_DIR = '{output_dir}'")


def preview_normalization():
    """Tampilkan preview normalisasi semua kelas dari kedua dataset."""
    print("\n📋 PREVIEW NORMALISASI NAMA KELAS")
    print("─" * 50)
    all_raw = set()
    for ds in DATASETS:
        if Path(ds['yaml']).exists():
            all_raw.update(load_class_names(ds['yaml']))

    mapping = {}
    for raw in sorted(all_raw):
        norm = normalize_class_name(raw)
        mapping.setdefault(norm, []).append(raw)

    print(f"\n{'Nama Standar':<25} {'Raw Names'}")
    print(f"{'─'*25} {'─'*35}")
    for norm, raws in sorted(mapping.items()):
        print(f"  {norm:<23} ← {', '.join(raws)}")
    print(f"\n  Total kelas standar: {len(mapping)}")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   YOLO → Keras Converter  (Multi-Dataset + Merge)   ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Preview normalisasi sebelum mulai
    preview_normalization()

    # Konfirmasi sebelum hapus folder lama
    if Path(OUTPUT_DIR).exists():
        print(f"\n🗑️  Menghapus folder lama: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir(parents=True)

    total_all = 0
    for ds in DATASETS:
        total_all += process_dataset(ds, OUTPUT_DIR, MARGIN)

    print(f"\n{'═'*60}")
    print(f"✅ Semua dataset selesai dikonversi! Total: {total_all} crop")

    print_summary(OUTPUT_DIR)


if __name__ == '__main__':
    main()
