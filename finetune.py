"""
Ogrenci Davranis Siniflandirma Modeli Egitimi
============================================
OgrenciHali klasorundeki resimleri kullanarak YOLOv8-cls modelini
2 sinif (Dinliyor / Dinlemiyor) uzerinde fine-tune eder.
"""

import os
import shutil
import random


def convert_heic_to_jpg(source_dir: str) -> None:
    """source_dir altindaki tum .heic dosyalarini .jpg'ye cevirir."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        print("pillow-heif yuklu degil, HEIC donusumu atlandi.")
        return

    from PIL import Image

    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".heic"):
                heic_path = os.path.join(root, f)
                jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
                if not os.path.exists(jpg_path):
                    img = Image.open(heic_path)
                    img.save(jpg_path, "JPEG")
                    print(f"Donusturuldu: {heic_path} -> {jpg_path}")


def split_dataset(source_dir: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42) -> None:
    random.seed(seed)
    classes = sorted(
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    )
    print(f"Siniflar: {classes}")

    for split in ("train", "val"):
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for img in train_imgs:
            shutil.copy2(
                os.path.join(cls_dir, img),
                os.path.join(output_dir, "train", cls, img),
            )
        for img in val_imgs:
            shutil.copy2(
                os.path.join(cls_dir, img),
                os.path.join(output_dir, "val", cls, img),
            )

        print(f"  {cls}: {len(train_imgs)} train, {len(val_imgs)} val")


def train_model(
    dataset_dir: str,
    base_model: str = "yolov8n-cls.pt",
    epochs: int = 10,
    imgsz: int = 224,
    project: str = "Ogrenci_Modeli",
) -> str:
    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(
        data=dataset_dir,
        epochs=epochs,
        imgsz=imgsz,
        project=project,
        batch=16,
        patience=10,
        device="cpu",
        verbose=True,
    )
    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    print("\nEgitim tamamlandi")
    print(f"En iyi model: {best_weights}")
    return best_weights


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DIR = os.path.join(BASE_DIR, "OgrenciHali")
    DATASET_DIR = os.path.join(BASE_DIR, "Ogrenci_Dataset")

    print("=" * 50)
    print("1) HEIC -> JPG donusumu")
    print("=" * 50)
    convert_heic_to_jpg(SOURCE_DIR)

    print("\n" + "=" * 50)
    print("2) Veri seti bolme (train/val)")
    print("=" * 50)
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        print("Eski Ogrenci_Dataset silindi.")
    split_dataset(SOURCE_DIR, DATASET_DIR)

    print("\n" + "=" * 50)
    print("3) Model Egitimi (2 sinif: Dinliyor / Dinlemiyor)")
    print("=" * 50)
    best = train_model(DATASET_DIR)
    print(f"Kullanima hazir model: {best}")
