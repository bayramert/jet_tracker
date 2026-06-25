import cv2                                             # OpenCV: video okuma/yazma ve görüntü gösterme
from tkinter import Tk, filedialog
from ultralytics import YOLO                           # YOLO modelini içe aktar

model_path = r"C:\Users\USER\Desktop\jet_tracker\runs\detect\train2\weights\best.pt"  # En iyi ağırlıkların yolu
model = YOLO(model_path)                               # Eğitilmiş ağırlıkları yükle

root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Islenecek videoyu sec",
    initialdir=r"C:\Users\USER\Desktop\jet_tracker",
    filetypes=[
        ("Video dosyalari", "*.mp4 *.avi *.mov *.mkv *.wmv"),
        ("Tum dosyalar", "*.*"),
    ],
)
root.destroy()

if not video_path:
    raise SystemExit("Video secilmedi, program kapatildi.")

cap = cv2.VideoCapture(video_path)                     # Videoyu aç
if not cap.isOpened():                                 # Açılamadıysa (yol hatası / codec vs.)
    raise SystemExit(f"Video açılamadı: {video_path}") # Hata ver ve çık

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0               # Videonun FPS değeri; yoksa 25’e düş
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))          # Video genişliği (piksel)
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))         # Video yüksekliği (piksel)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")              # MP4 yazıcı için codec (mp4v)
out = cv2.VideoWriter("jet_tespit_out.mp4", fourcc, fps, (w, h))  # Çıkış video yazıcısı

show_ok = True                                        # Pencere gösterimi mümkün mü (GUI var mı) bayrağı

while True:                                           # Video kare kare işlenecek
    ret, frame = cap.read()                           # Sıradaki kareyi oku
    if not ret:                                       # Kare kalmadıysa (video bitti)
        break                                         # Döngüden çık

    results = model(frame, conf=0.25, verbose=False)  # YOLO ile tespit yap (min güven 0.25)
    annotated = results[0].plot()                     # Kutu/etiket çizili görüntüyü al
    out.write(annotated)                              # Çıktı videosuna yaz

    if show_ok:                                       # Eğer GUI destekleniyorsa
        try:
            window_name = "Jet Tespit"
            cv2.imshow(window_name, annotated)       # Canlı önizleme penceresi
            if cv2.waitKey(1) & 0xFF == ord('q'):    # 'q' ile erken çık
                break
            # Kullanıcı pencereyi kapattıysa döngüyü sonlandır
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <= 0:
                break
        except cv2.error:                              # Bazı ortamlar GUI desteklemez (ör. uzak sunucu)
            show_ok = False                           # Bir daha pencere açmayı deneme

cap.release()                                         # Video dosyasını serbest bırak
out.release()                                         # Çıkış video dosyasını kapat
cv2.destroyAllWindows()                               # Açık pencereleri kapat
print("Bitti. Çıktı: jet_tespit_out.mp4")             # Kullanıcıya bilgi ver
