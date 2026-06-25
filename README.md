# ✈️ Jet Tracker

YOLOv8 kullanılarak jet/uçak tespiti yapan proje.

## Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/kullaniciadi/jet_tracker.git
cd jet_tracker
```

### 2. Gerekli kütüphaneleri yükleyin

```bash
pip install ultralytics opencv-python requests pyyaml
```

## Kullanım

### Veri setini indir

```bash
python jet.py
```

Bu komut veri setini indirip `roboflow_dataset` klasörüne çıkarır.

### Modeli eğit

```bash
python fix_dataset_and_train.py
```

Eğitim tamamlandığında en iyi model:

```text
runs/detect/train*/weights/best.pt
```

altında oluşacaktır.

### Video üzerinde tespit yap

`play.py` dosyasındaki model yolunu kendi eğitim çıktınıza göre güncelleyin:

```python
model_path = r"runs\detect\train2\weights\best.pt"
```

Ardından:

```bash
python play.py
```

İşlemek istediğiniz videoyu seçin.

İşlem tamamlandığında çıktı:

```text
jet_tespit_out.mp4
```

olarak kaydedilir.

## Proje Dosyaları

| Dosya | Açıklama |
|---------|---------|
| `jet.py` | Veri setini indirir |
| `fix_dataset_and_train.py` | Veri setini düzenler ve modeli eğitir |
| `play.py` | Eğitilmiş model ile video üzerinde tespit yapar |
