# Ders Ogrenci Durum Takibi

Ogrencilerin dersteki dikkat/katilim durumunu goruntu uzerinden siniflandiran bir proje. YOLOv8 ile siniflandirma egitimi, Streamlit arayuzu, ve PDF'den sesli video uretimi icin ayri moduller icerir.

## Ozellikler
- Ogrenci durum siniflandirma (Dinliyor / Dinlemiyor)
- Streamlit tabanli arayuz (kamera, resim, PDF->video)
- PDF'den slayt + Turkce TTS ile video olusturma

## Proje Yapisi
- finetune.py: Veri seti bolme ve YOLOv8-cls egitimi
- streamlit_app.py: Arayuz ve analiz akisi
- pdf_to_video.py: PDF'den sesli video uretimi
- requirements.txt: Python bagimliliklari

## Kurulum
```bash
pip install -r requirements.txt
```

## Calistirma
### 1) Egitim
```bash
python finetune.py
```

### 2) Streamlit uygulamasi
```bash
streamlit run streamlit_app.py
```

### 3) PDF'den video
```bash
python pdf_to_video.py <pdf_dosyasi>
```

## Notlar
- Dataset, modeller ve cikti dosyalari repo disinda tutulur (bkz. .gitignore).
- Model agirliklarini (or. yolov8n.pt, yolov8n-cls.pt) calistirmadan once proje klasorune koymalisin.
