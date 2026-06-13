import os
import sys
import zipfile
import urllib.request
from pathlib import Path


REPO_ZIP_URL = (
    "https://github.com/Stormynova/Uno_Card_detection/archive/refs/heads/main.zip"
)
ZIP_PATH    = "uno_dataset.zip"
EXTRACT_DIR = "."
DATASET_DIR = "dataset"


def download_with_progress(url: str, dest: str):
    print(f"⬇️  Mengunduh dari:\n   {url}\n")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            mb_done  = downloaded / 1_048_576
            mb_total = total_size  / 1_048_576
            print(f"\r   [{bar}] {pct:.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print("\n✅ Unduhan selesai.")


def extract_and_organize(zip_path: str, dataset_dir: str):
    print(f"\n📦 Mengekstrak {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("✅ Ekstraksi selesai.")

    # Cari folder berisi sub-folder kelas kartu
    extracted_root = Path("Uno_Card_detection-main")
    if not extracted_root.exists():
        print("⚠️  Folder hasil ekstraksi tidak sesuai ekspektasi.")
        print("   Cari manual folder yang berisi sub-folder kartu dan simpan sebagai 'dataset/'")
        return

    # Cari sub-folder mana yang berisi gambar
    candidate_dirs = []
    for p in extracted_root.rglob("*"):
        if p.is_dir():
            imgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg"))
            if len(imgs) > 0:
                candidate_dirs.append(p.parent)
                break

    if candidate_dirs:
        source = candidate_dirs[0]
        print(f"\n📁 Folder kelas ditemukan di: {source}")

        # Pindahkan ke 'dataset/'
        if not Path(dataset_dir).exists():
            import shutil
            shutil.copytree(str(source), dataset_dir)
            print(f"✅ Dataset disalin ke ./{dataset_dir}/")
        else:
            print(f"   Folder '{dataset_dir}' sudah ada, dilewati.")
    else:
        print("⚠️  Tidak dapat menemukan folder gambar secara otomatis.")
        print(f"   Salin manual folder kelas kartu ke './{dataset_dir}/'")


def verify_dataset(dataset_dir: str):
    path = Path(dataset_dir)
    if not path.exists():
        print(f"\n❌ Folder '{dataset_dir}' tidak ditemukan!")
        return False

    classes = [d.name for d in sorted(path.iterdir()) if d.is_dir()]
    if not classes:
        print(f"\n❌ Tidak ada sub-folder kelas di '{dataset_dir}'!")
        return False

    total_images = 0
    print(f"\n📊 Verifikasi Dataset ({dataset_dir}/):")
    print(f"   {'Kelas':<25} {'Gambar':>8}")
    print(f"   {'─'*25} {'─'*8}")
    for cls in classes:
        imgs = list((path / cls).glob("*.jpg")) + \
               list((path / cls).glob("*.jpeg")) + \
               list((path / cls).glob("*.png"))
        total_images += len(imgs)
        print(f"   {cls:<25} {len(imgs):>8}")

    print(f"\n   Total: {len(classes)} kelas, {total_images} gambar")
    return True


def main():
    print("╔══════════════════════════════════════════╗")
    print("║   UNO Dataset Setup — Stormynova/GitHub  ║")
    print("╚══════════════════════════════════════════╝\n")

    # Cek apakah dataset sudah ada
    if Path(DATASET_DIR).exists():
        print(f"ℹ️  Folder '{DATASET_DIR}' sudah ada.")
        if verify_dataset(DATASET_DIR):
            print("\n✅ Dataset siap digunakan. Jalankan: python train.py")
            return

    # Unduh jika belum ada ZIP
    if not Path(ZIP_PATH).exists():
        try:
            download_with_progress(REPO_ZIP_URL, ZIP_PATH)
        except Exception as e:
            print(f"\n❌ Gagal mengunduh: {e}")
            print("\nCara alternatif:")
            print("  1. Kunjungi https://github.com/Stormynova/Uno_Card_detection")
            print("  2. Klik 'Code' → 'Download ZIP'")
            print(f"  3. Ekstrak dan salin folder kelas ke './{DATASET_DIR}/'")
            sys.exit(1)

    # Ekstrak dan organisir
    extract_and_organize(ZIP_PATH, DATASET_DIR)

    # Verifikasi
    if verify_dataset(DATASET_DIR):
        print("\n🎉 Setup selesai! Jalankan: python train.py")

        # Hapus ZIP untuk menghemat ruang
        os.remove(ZIP_PATH)
        print("🗑️  File ZIP dihapus untuk menghemat ruang disk.")
    else:
        print("\n⚠️  Verifikasi gagal. Periksa struktur folder dataset secara manual.")


if __name__ == '__main__':
    main()
