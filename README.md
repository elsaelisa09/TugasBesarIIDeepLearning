# SmartFace - Sistem Absensi Face Recognition

Tugas Besar II Deep Learning - Kelas RA

## Anggota Kelompok

| Nama                       | NIM       |
| -------------------------- | --------- |
| Elsa Elisa Yohana Sianturi | 122140135 |
| Sikah Nubuahtul Ilmi       | 122140208 |
| Machzaul Harmansyah        | 122140172 |

## Deskripsi

Sistem absensi berbasis face recognition menggunakan ResNet50 fine-tuned dengan ArcFace loss untuk mengenali 70 identitas mahasiswa. Backend menggunakan Flask + PyTorch dengan MTCNN untuk deteksi wajah, frontend menggunakan React + TypeScript dengan Tailwind CSS dan shadcn/ui components.

## Demo & Deployment

### 🚀 Live Application

- **Frontend**: Deployed on **Vercel** - [https://tugas-besar-ii-deep-learning.vercel.app](https://tugas-besar-ii-deep-learning.vercel.app)
- **Backend API**: Deployed on **Hugging Face Spaces** - [https://huggingface.co/spaces/elsaelisa09/smartface-backend](https://huggingface.co/spaces/elsaelisa09/smartface-backend)
- **Model Hub**: [https://huggingface.co/elsaelisa09/smartface-attendance-model](https://huggingface.co/elsaelisa09/smartface-attendance-model)

### 📦 Platform yang Digunakan

**Vercel** (Frontend)

- Hosting React + Vite application
- Auto-deploy dari GitHub repository
- Build otomatis setiap push ke branch main

**Hugging Face Spaces** (Backend)

- Hosting Flask API dengan Docker
- Model PyTorch ResNet50 + ArcFace
- Model weights dari Hugging Face Hub

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

## Cara Deploy

### Deploy ke Vercel (Frontend)

1. Push code ke GitHub repository
2. Login ke [vercel.com](https://vercel.com) dan import repository
3. Set root directory ke `SmartFace`
4. Framework akan terdeteksi otomatis (Vite)
5. Deploy - Done! ✨

### Deploy ke Hugging Face Spaces (Backend)

1. Login ke [huggingface.co](https://huggingface.co)
2. Create New Space dengan Docker SDK
3. Push code dari folder `hf-space` atau `backend`
4. Include `Dockerfile`, `app.py`, `requirements.txt`
5. Model akan auto-load dari Hugging Face Hub
6. Space akan build dan running otomatis 🚀

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

ResNet50 fine-tuned dengan ArcFace loss untuk 70 classes (identitas mahasiswa). Menggunakan MTCNN untuk face detection, input 224x224 RGB.

## Fitur

- 📸 Real-time face detection & recognition
- ✅ Automatic attendance marking
- 📊 Attendance history & management
- 🎯 High accuracy dengan ArcFace loss
- 📱 Responsive modern UI

## Referensi

- He, K., et al. (2016). Deep Residual Learning for Image Recognition
- Zhang, K., et al. (2016). Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

Institut Teknologi Sumatera - 2025
