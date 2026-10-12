import os

# Olası kütüphane çakışmalarını (OpenMP Error #15) engellemek için güvenlik satırı
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import glob

# --- AYARLAR ---
# Sisli VisDrone görsellerimizin bulunduğu tam yol
IMAGE_FOLDER = r"C:\Users\KAAN\PycharmProjects\Bytetrack\ByteTrack\datasets\VIS-MOT17-fog-3"

# Çıktı video adını test komutumuzla uyumlu olacak şekilde ayarladık
OUTPUT_VIDEO = "VisDrone_Sisli_Test-3.avi"
FPS = 30


def images_to_video():
    # Klasördeki resimleri al ve sırala (0001.jpg, 0002.jpg...)
    images = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    # Eğer .jpg dışında (örn .png) varsa onları da dahil etmek için alternatif:
    if not images:
        images = glob.glob(os.path.join(IMAGE_FOLDER, "*.*"))
        images = [img for img in images if img.endswith(('.jpg', '.jpeg', '.png'))]

    images.sort()

    if not images:
        print(f"HATA: '{IMAGE_FOLDER}' içinde resim bulunamadı!")
        print("Lütfen yolun doğru olduğundan emin ol.")
        return

    # İlk resimden boyutları öğren
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Video Kaydediciyi Başlat (Derin öğrenme modelleri için en güvenli XVID formatı)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

    print(f"Video oluşturuluyor: {len(images)} kare işlenecek...")

    for image_path in images:
        video.write(cv2.imread(image_path))

    video.release()
    cv2.destroyAllWindows()
    print(f"BAŞARILI! Video kaydedildi: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    images_to_video()