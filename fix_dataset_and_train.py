import os, random, shutil, yaml                # OS işlemleri, rastgele seçim, dosya kopyalama, YAML okuma/yazma için
from pathlib import Path                       # Yolları platformdan bağımsız yönetmek için Path

# ---- KURULUM ----
random.seed(42)                                # Bölme işleminin tekrarlanabilir olması için rastgelelik sabitlenir
ROOT = Path(r"C:\Users\USER\Desktop\jet_tracker\roboflow_dataset")  # Dataset kök klasörü (ZIP'ten çıkan)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}   # Desteklenen görüntü uzantıları

# ---- KLASÖRLER ----
train_img = ROOT/"train"/"images"              # train görüntü dizini
train_lbl = ROOT/"train"/"labels"              # train label dizini
valid_img = ROOT/"valid"/"images"              # valid görüntü dizini
valid_lbl = ROOT/"valid"/"labels"              # valid label dizini
test_img  = ROOT/"test"/"images"               # test görüntü dizini
test_lbl  = ROOT/"test"/"labels"               # test label dizini

# Eksikse oluştur
for d in [valid_img, valid_lbl, test_img, test_lbl]:  # valid/test klasörlerinin varlığını garanti eder
    d.mkdir(parents=True, exist_ok=True)              # Yoksa oluştur; varsa hata verme

# ---- GÖRSELLERİ TOPLA ----
images = [p for p in train_img.rglob("*") if p.suffix.lower() in IMG_EXTS]  # train/images altında tüm görselleri (alt klasörler dahil) topla
if not images:                                         # Hiç görüntü bulunamadıysa
    raise SystemExit(f"Train klasöründe görüntü bulunamadı: {train_img}")   # Kullanıcıyı durdurucu hata mesajı ile bilgilendir

# Eğer valid/test zaten DOLUySA tekrar bölme (kopya çoğaltmayı önler)
valid_already = any(valid_img.rglob("*"))              # valid/images altında herhangi bir dosya/klasör var mı?
test_already  = any(test_img.rglob("*"))               # test/images altında herhangi bir dosya/klasör var mı?

# Sadece eksikse böl
if (not valid_already) or (not test_already):          # valid veya test boşsa yeniden bölme yapılacak
    random.shuffle(images)                              # Görselleri karıştır (seed sayesinde deterministik)
    n = len(images)                                     # Toplam train görsel sayısı
    n_valid = max(1, int(0.20 * n))                     # %20 valid boyutu, en az 1
    n_test  = max(1, int(0.10 * n))                     # %10 test boyutu, en az 1

    valid_set = set(images[:n_valid])                   # İlk %20’lik kısmı valid set
    test_set  = set(images[n_valid:n_valid+n_test])     # Sonraki %10’luk kısmı test set

    def copy_pair(img_path: Path, dst_img_dir: Path, dst_lbl_dir: Path):
        dst_img_dir.mkdir(parents=True, exist_ok=True)  # Hedef görüntü klasörü yoksa oluştur
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)  # Hedef label klasörü yoksa oluştur
        shutil.copy2(img_path, dst_img_dir / img_path.name)  # Görüntüyü hedefe kopyala (zaman damgaları korunur)
        lbl_name = img_path.with_suffix(".txt").name    # YOLO etiket dosyası isim kuralı: aynı ad + .txt
        src_lbl = train_lbl / lbl_name                  # Etiketin train/labels içindeki beklenen tam yolu
        if src_lbl.exists():                            # Etiket varsa (bazı görseller boş/etiketsiz olabilir)
            shutil.copy2(src_lbl, dst_lbl_dir / lbl_name)  # Etiketi hedef labels klasörüne kopyala

    if not valid_already:                               # valid boşsa valid’e kopyala
        for img in valid_set:
            copy_pair(img, valid_img, valid_lbl)
    if not test_already:                                # test boşsa test’e kopyala
        for img in test_set:
            copy_pair(img, test_img, test_lbl)

    print(f"Toplam train görüntü: {len(images)} | valid: {len(list(valid_img.glob('*')))} | test: {len(list(test_img.glob('*')))}")
else:
    print("valid/ ve test/ zaten dolu; bölme atlandı.") # Eğer doluysa ikinci kez çoğaltmayı engeller

# ---- data.yaml'ı GÜNCELLE ----
yaml_path = ROOT/"data.yaml"                            # data.yaml tam yolu
with open(yaml_path, "r", encoding="utf-8") as f:      # YAML dosyasını oku
    data = yaml.safe_load(f)                            # İçeriği Python dict olarak al

# Yol alanlarını normalize et (göreli yollar yeterli)
data["train"] = "train/images"                          # YOLO için train görüntü yolu
data["val"]   = "valid/images"                          # YOLO için val görüntü yolu
data["test"]  = "test/images"                           # YOLO için test görüntü yolu

with open(yaml_path, "w", encoding="utf-8") as f:      # YAML dosyasını tekrar aç (yazma)
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)  # Değişiklikleri kaydet (Türkçe karakterlere izin ver)

print("data.yaml güncellendi ->", yaml_path)            # Kullanıcıya bilgi ver

# ---- YOLO EĞİTİMİ ----
from ultralytics import YOLO                            # Ultralytics YOLO paketinden sınıfı içe aktar
model = YOLO("yolov8n.pt")                              # Hazır küçük model (nano) ağırlıkları ile başlangıç
results = model.train(                                  # Eğitim başlat
    data=str(yaml_path),                                # data.yaml yolu
    epochs=25,                                          # Eğitim epoch sayısı
    imgsz=640,                                          # Girdi görüntü boyutu
    workers=0,   # Windows + CPU için stabil                     # DataLoader worker sayısı (Windows’ta 0 önerilir)
    batch=8      # CPU için güvenli                           # Batch size (CPU’da düşük tutmak stabilite sağlar)
)
print("Eğitim bitti. Çıktılar:", results.save_dir)      # Eğitim klasörü (runs/detect/trainX) bilgisini yazdır
