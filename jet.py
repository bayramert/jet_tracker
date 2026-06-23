import requests, zipfile, io, os                      # HTTP indirme, ZIP açma ve dosya işlemleri için

url = "https://app.roboflow.com/ds/UAcaz0VT5j?key=hf6K82WerL"  # Roboflow kısa süreli indirme URL’si (anahtar içerir)
zip_path = "roboflow.zip"                              # ZIP dosyasının yerel adı

# ZIP indir
r = requests.get(url)                                  # URL’den içerik çek
with open(zip_path, "wb") as f:                        # ZIP’i yerel olarak aç/yaz
    f.write(r.content)                                 # İndirilen baytları diske yaz

# ZIP aç
with zipfile.ZipFile(zip_path, 'r') as zip_ref:        # ZIP dosyasını aç
    zip_ref.extractall("roboflow_dataset")             # İçeriği roboflow_dataset klasörüne çıkar

# ZIP sil
os.remove(zip_path)                                    # Geçici ZIP dosyasını temizle

print("Dataset indirildi ve 'roboflow_dataset' klasörüne açıldı.")  # Kullanıcıya bilgi

from ultralytics import YOLO                           # YOLO’yu içe aktar
model = YOLO("yolov8n.pt")                             # Nano modelle başla (hızlı/az parametreli)
data_yaml_tam_yolu = 'roboflow_dataset/data.yaml'      # İndirilen datasetin data.yaml yolu
results = model.train(data=data_yaml_tam_yolu, epochs=25, imgsz=640)  # Basit eğitim çağrısı (varsayılan diğer ayarlar)

