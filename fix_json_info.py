import json
import os
import glob

# Annotations klasöründeki TÜM .json dosyalarını bul (train.json, val.json vs.)
json_files = glob.glob("datasets/MOT17/annotations/*.json")

if not json_files:
    print("HATA: Hiçbir json dosyası bulunamadı! Yolunu kontrol et.")
else:
    print(f"Bulunan dosya sayısı: {len(json_files)}")

    for json_path in json_files:
        print(f"İşleniyor: {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 'info' anahtarı yoksa ekle
            if 'info' not in data:
                data['info'] = {
                    "description": "MOT17 Foggy Dataset",
                    "url": "",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "Tez Calismasi",
                    "date_created": "2025-12-04"
                }

                # Dosyayı tekrar kaydet
                with open(json_path, 'w') as f:
                    json.dump(data, f)
                print(" -> BAŞARILI: 'info' eklendi ve kaydedildi.")
            else:
                print(" -> Zaten 'info' var, işlem yapılmadı.")

        except Exception as e:
            print(f" -> HATA OLUŞTU: {e}")

print("\n--- TÜM İŞLEMLER TAMAMLANDI ---")