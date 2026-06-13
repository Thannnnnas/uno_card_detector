import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

def prepare_data(data_dir: str, img_size: int = 224, batch_size: int = 32):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {data_dir}")

    valid_classes = []
    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir():
            continue
        images = [f for f in folder.iterdir()
                  if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
        if len(images) == 0:
            print(f"  ⚠️  Folder kosong dilewati: {folder.name}")
            continue
        valid_classes.append(folder.name)

    if len(valid_classes) == 0:
        raise ValueError("Tidak ada folder kelas valid yang ditemukan di dataset!")

    print(f"\n✅ Ditemukan {len(valid_classes)} kelas:")
    print(", ".join(valid_classes))

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        shear_range=0.15,
        fill_mode='nearest',
        validation_split=0.2   
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    print(f"\n📂 Memuat data dari: {data_dir}")

    # Training: 80% dari total
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=99
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    print(f"\n📊 Statistik Dataset:")
    print(f"   Training  samples : {train_gen.samples}")
    print(f"   Validation samples: {val_gen.samples}")
    print(f"   Jumlah kelas      : {num_classes}")

    return train_gen, val_gen, test_gen, class_names

def build_model(num_classes: int, img_size: int = 224, learning_rate: float = 0.001):
    print("\n🔧 Membangun arsitektur MobileNetV2...")

    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False
    print(f"   Base model layers  : {len(base_model.layers)}")
    print(f"   Frozen layers      : {sum(1 for l in base_model.layers if not l.trainable)}")

    x = base_model.output
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Kompilasi model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def get_callbacks(checkpoint_path: str, patience_early: int = 10, patience_lr: int = 3):
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=patience_early,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            verbose=1,
            min_lr=1e-6
        )
    ]
    return callbacks


def train_phase1(model, train_gen, val_gen, epochs: int = 40):
    print("\n" + "="*60)
    print("🚀 FASE 1 — Feature Extraction (Base Frozen)")
    print("="*60)

    callbacks = get_callbacks('uno_mobilenetv2_phase1.h5')

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✅ Fase 1 selesai. Model terbaik disimpan ke 'uno_mobilenetv2_phase1.h5'")
    return history1


def train_phase2(model, train_gen, val_gen, epochs: int = 40,
                 unfreeze_layers: int = 50):
    print("\n" + "="*60)
    print("🔥 FASE 2 — Fine-Tuning (Unfreeze 30 layer terakhir)")
    print("="*60)

    # Unfreeze layer terakhir dari base model
    base_model = model.layers[0] if hasattr(model.layers[0], 'layers') else None

    # Cari layer MobileNetV2 (layer pertama yang berupa Model)
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_layer = layer
            break

    if base_layer is None:
        # Fallback: unfreeze dari model langsung
        for layer in model.layers[-unfreeze_layers:]:
            layer.trainable = True
    else:
        # Bekukan semua dulu, lalu unfreeze 30 terakhir
        base_layer.trainable = True
        for layer in base_layer.layers[:-unfreeze_layers]:
            layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"   Layer yang bisa dilatih: {trainable_count}")

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = get_callbacks('uno_mobilenetv2_final.h5')

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n✅ Fase 2 selesai. Model final disimpan ke 'uno_mobilenetv2_final.h5'")
    return history2

def evaluate_model(model, test_gen, class_names: list):
    print("\n" + "="*60)
    print("📈 EVALUASI MODEL")
    print("="*60)

    test_gen.reset()
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\n🎯 Test Accuracy : {accuracy * 100:.2f}%")
    print(f"   Test Loss     : {loss:.4f}")

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = test_gen.classes[:len(y_pred)]

    labels_used = sorted(set(y_true) | set(y_pred))
    target_names = [class_names[i] for i in labels_used]

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred,
                                labels=labels_used,
                                target_names=target_names))

    cm = confusion_matrix(y_true, y_pred, labels=labels_used)
    _plot_confusion_matrix(cm, target_names)

    return accuracy, loss, y_true, y_pred


def _plot_confusion_matrix(cm: np.ndarray, class_names: list,
                            save_path: str = 'confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(max(12, len(class_names)), max(10, len(class_names) - 2)))

    # Normalisasi per baris (recall per kelas)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix — UNO Card Classifier (MobileNetV2)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n💾 Confusion matrix disimpan ke: {save_path}")


def plot_training_history(history1, history2, save_path: str = 'training_history.png'):
    acc  = history1.history['accuracy']  + history2.history['accuracy']
    val_acc  = history1.history['val_accuracy']  + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
    split_epoch  = len(history1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — UNO MobileNetV2', fontsize=14, y=1.02)

    # ─ Akurasi ─
    ax1.plot(epochs_range, acc,     label='Train Accuracy',      color='royalblue')
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy', color='coral', linestyle='--')
    ax1.axvline(x=split_epoch, color='gray', linestyle=':', linewidth=1.5, label='Phase 1→2')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ─ Loss ─
    ax2.plot(epochs_range, loss,     label='Train Loss',      color='royalblue')
    ax2.plot(epochs_range, val_loss, label='Validation Loss', color='coral', linestyle='--')
    ax2.axvline(x=split_epoch, color='gray', linestyle=':', linewidth=1.5, label='Phase 1→2')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Training history plot disimpan ke: {save_path}")

def predict_uno_card(image_path: str, model, class_names: list,
                     img_size: int = 224):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_batch, verbose=0)
    class_idx   = np.argmax(predictions[0])
    confidence  = predictions[0][class_idx] * 100.0
    predicted_class = class_names[class_idx]

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediksi: {predicted_class}\nKepercayaan: {confidence:.1f}%",
              fontsize=13, color='darkblue', fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\n🃏 Prediksi  : {predicted_class}")
    print(f"   Confidence: {confidence:.2f}%")

    return predicted_class, confidence


def predict_from_camera(model, class_names: list, img_size: int = 224):
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV tidak terinstal. Jalankan: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Tidak dapat membuka kamera.")
        return

    print("📷 Kamera aktif. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized   = cv2.resize(rgb_frame, (img_size, img_size))
        normalized = resized.astype('float32') / 255.0
        batch     = np.expand_dims(normalized, axis=0)

        # Prediksi
        preds      = model.predict(batch, verbose=0)
        idx        = np.argmax(preds[0])
        confidence = preds[0][idx] * 100
        label      = class_names[idx]

        # Overlay teks di frame
        text = f"{label}: {confidence:.1f}%"
        cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

        cv2.imshow('UNO Card Classifier — Tekan Q untuk keluar', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("📷 Kamera ditutup.")


def main():
    DATA_DIR   = '/root/kholid/uno/keras_dataset'   
    IMG_SIZE   = 224
    BATCH_SIZE = 16

    # STEP 1: Persiapan data
    print("\n" + "="*60)
    print("📦 STEP 1 — Persiapan Data")
    print("="*60)
    train_gen, val_gen, test_gen, class_names = prepare_data(
        DATA_DIR, IMG_SIZE, BATCH_SIZE
    )
    num_classes = len(class_names)

    # Simpan daftar kelas
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"💾 class_names.json disimpan ({num_classes} kelas)")

    # STEP 2: Bangun model
    print("\n" + "="*60)
    print("🏗️  STEP 2 — Arsitektur Model")
    print("="*60)
    model = build_model(num_classes, IMG_SIZE, learning_rate=0.001)

    # STEP 3: Training
    history1 = train_phase1(model, train_gen, val_gen, epochs=40)
    history2 = train_phase2(model, train_gen, val_gen, epochs=40)

    # STEP 4: Evaluasi
    accuracy, loss, y_true, y_pred = evaluate_model(model, test_gen, class_names)
    plot_training_history(history1, history2, save_path='training_history.png')

    # Ringkasan akhir
    print("\n" + "="*60)
    print("🏁 TRAINING SELESAI")
    print("="*60)
    print(f"   Test Accuracy : {accuracy*100:.2f}%")
    print(f"   Test Loss     : {loss:.4f}")
    print(f"   Model         : uno_mobilenetv2_final.h5")
    print(f"   Class Names   : class_names.json")
    print(f"   History Plot  : training_history.png")
    print(f"   Confusion Mat : confusion_matrix.png")

    if accuracy >= 0.90:
        print("\n🎉 Target akurasi ≥90% TERCAPAI!")
    else:
        print(f"\n⚠️  Akurasi {accuracy*100:.1f}% belum mencapai target 90%.")
        print("   Saran: tambah epochs, kurangi dropout, atau perbanyak data.")


if __name__ == '__main__':
    main()
