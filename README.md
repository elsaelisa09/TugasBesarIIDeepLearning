# SmartFace - Sistem Absensi Face Recognition

Tugas Besar II Deep Learning - Kelas RA

## Anggota Kelompok

| Nama                       | NIM       |
| -------------------------- | --------- |
| Elsa Elisa Yohana Sianturi | 122140135 |
| Sikah Nubuahtul Ilmi       | 122140208 |
| Machzaul Harmansyah        | 122140172 |

## Deskripsi

Sistem absensi berbasis face recognition menggunakan ResNet50 fine-tuned untuk mengenali 70 identitas mahasiswa. Backend menggunakan Flask + PyTorch dengan MTCNN untuk deteksi wajah, frontend menggunakan React + TypeScript.

## Teknologi

**Backend**: Python, PyTorch 2.8.0, Flask 3.1.0, OpenCV 4.12.0, MTCNN

**Frontend**: React 18.3.1, TypeScript 5.8.3, Vite 5.4.19, Tailwind CSS

## Instalasi

```bash
# Clone repository
git clone https://github.com/elsaelisa09/TugasBesarIIDeepLearning.git
cd TugasBesarIIDeepLearning/SmartFace

# Setup backend
cd backend
pip install -r requirements.txt

# Setup frontend
cd ..
npm install
```

## Menjalankan Aplikasi

```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
npm run dev
```

Backend: `http://localhost:5000`
Frontend: `http://localhost:8080`

## Struktur Project

```
SmartFace/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── best_finetuned_resnet50.pth
│   └── attendance.json
└── src/
    ├── components/
    ├── pages/
    └── main.tsx
```

## Model

ResNet50 fine-tuned dengan 70 classes, input 224x224 RGB. Menggunakan MTCNN untuk face detection.

## Referensi

- He, K., et al. (2016). Deep Residual Learning for Image Recognition
- Zhang, K., et al. (2016). Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

Institut Teknologi Sumatera - 2025
