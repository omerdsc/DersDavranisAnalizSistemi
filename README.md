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
<img width="1916" height="909" alt="Ekran görüntüsü 2026-03-04 213538" src="https://github.com/user-attachments/assets/20bc029f-f434-4643-8b66-b8e6d6c61dc3" />

<img width="1551" height="770" alt="Ekran-goruntusu-2026-02-10-104113" src="https://github.com/user-attachments/assets/7704f429-3684-4f49-9b2f-ad437cde698a" />

<img width="1220" height="619" alt="Ekran-goruntusu-2026-02-10-104023" src="https://github.com/user-attachments/assets/248b6b55-f129-4a40-b4b2-a3a5be6c9669" />



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
