import os
import cv2
import numpy as np
from tqdm import tqdm

# --- TAM VE KESİN DOSYA YOLLARI ---
SRC_DIR = r"C:\Users\KAAN\PycharmProjects\Bytetrack\ByteTrack\datasets\MOT17_clean\test\VIS-MOT17-3"
DST_DIR = r"C:\Users\KAAN\PycharmProjects\Bytetrack\ByteTrack\datasets\VIS-MOT17-fog-3"

SIS_YOGUNLUGU = 0.4  # Makaledeki Yoğun Sis derecesi


def add_fog(image, beta=0.05, brightness=0.7):
    """Tek bir resme sis efekti ekler"""
    row, col, chs = image.shape
    # Basit derinlik haritası (Üst taraf uzak, alt taraf yakın)
    x = np.linspace(0, 1, row)
    dist = np.tile(x.reshape(row, 1), (1, col))
    d = 1.0 - dist  # Derinliği ters çevir

    # Sis formülü
    transmission = np.exp(-beta * d * 8)
    transmission = transmission[:, :, np.newaxis]
    A = 255 * brightness

    foggy_image = image * transmission + A * (1 - transmission)
    return np.array(foggy_image, dtype=np.uint8)


def main():
    if not os.path.exists(SRC_DIR):
        print(f"HATA: Kaynak klasör bulunamadı:\n{SRC_DIR}")
        return

    # Hedef klasörü oluştur (Daha önce oluştuysa bile içine yazmaya devam eder)
    os.makedirs(DST_DIR, exist_ok=True)

    # Klasördeki sadece resim dosyalarını (.jpg, .png) listeye al
    images = [f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"HATA: {SRC_DIR} içinde hiç görsel bulunamadı!")
        return

    print(f"Toplam {len(images)} görsel bulundu. Sisleme işlemi başlıyor...")

    # İlerleme çubuğu ile tüm resimleri tek tek işle
    for img_name in tqdm(images, desc="Sis Ekleniyor"):
        src_path = os.path.join(SRC_DIR, img_name)
        dst_path = os.path.join(DST_DIR, img_name)

        # Resmi Oku
        img = cv2.imread(src_path)
        if img is None:
            print(f"\nUyarı: Görsel okunamadı, atlanıyor -> {img_name}")
            continue

        # Sis Ekle
        foggy_img = add_fog(img, beta=SIS_YOGUNLUGU)

        # Sisli halini kaydet
        cv2.imwrite(dst_path, foggy_img)

    print("\n--- İŞLEM TAMAMLANDI ---")
    print(f"Bütün sisli görseller buraya başarıyla kaydedildi:\n{DST_DIR}")


if __name__ == "__main__":
    main()