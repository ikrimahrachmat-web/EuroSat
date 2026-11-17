# Klasifikasi Citra EuroSAT

Aplikasi ini melatih CNN dari citra RGB EuroSAT. Pengguna dapat bermain kuis
tebak kelas dari galeri citra, mengunggah satu atau beberapa gambar untuk
melihat prediksi dan ringkasan kelas, mengunduh hasil analisis sebagai CSV,
atau membaca deskripsi kelas. Prediksi hanya sesuai untuk citra satelit sejenis
data latih, bukan foto ponsel biasa.

## 1. Siapkan dataset

Unduh dan ekstrak bagian RGB EuroSAT. Pastikan folder yang dipilih langsung
berisi satu subfolder untuk setiap kelas, contohnya:

```text
data/
└── EuroSAT/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    ├── Highway/
    ├── Industrial/
    ├── Pasture/
    ├── PermanentCrop/
    ├── Residential/
    ├── River/
    └── SeaLake/
```

Jika hasil ekstraksi memiliki folder tambahan di dalam `EuroSAT`, pilih
folder yang tepat melalui kolom lokasi dataset di panel aplikasi.

## 2. Instalasi dan menjalankan

Gunakan Python 3.11 atau 3.12, lalu jalankan dari folder proyek:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Di aplikasi, masukkan lokasi folder dataset, lalu klik **Mulai pelatihan**.
Setelah pelatihan selesai, gunakan tab **Tebak Citra** untuk kuis dari contoh
dataset, tab **Upload Gambar** untuk menguji satu gambar, tab **Analisis Banyak
Gambar** untuk menganalisis beberapa gambar sekaligus dan mengunduh ringkasan
CSV, atau tab **Kenali Kelas** untuk membaca deskripsi kategori EuroSAT.

Model dan daftar kelas disimpan di folder `models/` agar dapat digunakan
kembali tanpa melatih ulang.
