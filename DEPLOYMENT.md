# 🚀 Deployment Guide - SmartFace Attendance System

Panduan lengkap untuk deploy aplikasi Face Recognition Attendance ke production.

## 📋 Prerequisites

- Akun GitHub (untuk repository)
- Akun Vercel (untuk frontend) - https://vercel.com
- Akun Railway/Render/Koyeb (untuk backend) - pilih salah satu:
  - Railway: https://railway.app (Recommended - $5/month)
  - Render: https://render.com (Free tier tersedia)
  - Koyeb: https://koyeb.com (Free tier tersedia)

---

## 🎯 Architecture Overview

```
Frontend (React + Vite)  →  Backend (Flask + PyTorch)
   [Vercel]                    [Railway/Render]
                                    ↓
                            Model: best_gacor.pth
                            Face Detection: MTCNN
```

---

## 📦 Part 1: Backend Deployment (Railway/Render)

### ✅ Model Already on Hugging Face!

Model `best_gacor.pth` sudah diupload ke:
- **URL:** https://huggingface.co/elsaelisa09/smartface-attendance-model
- **Size:** 94.16 MB
- **Status:** Public (dapat diakses tanpa token)

Backend akan otomatis download model saat pertama kali deploy! 🚀

### Step 1: Prepare Repository

1. **Commit semua perubahan ke Git:**
```powershell
cd "C:\Users\elsae\.vscode\Documents\Dokumentasi Kuliah\SEMESTER VII\TugasBesarIIDeepLearning"
git add .
git commit -m "Prepare for deployment with Hugging Face model"
git push origin elsaaaa
```

**Note:** Model `.pth` file tidak akan ter-commit (sudah di `.gitignore`)

### Step 2A: Deploy ke Railway (Trial $5)

1. **Login ke Railway:** https://railway.app
2. **Create New Project** → Deploy from GitHub repo
3. **Select repository:** `elsaelisa09/TugasBesarIIDeepLearning`
4. **Root Directory:** Pilih `SmartFace/backend`
5. **Configure Environment Variables:**
   - Klik tab "Variables"
   - Add variables:
     ```
     FLASK_ENV=production
     USE_HUGGINGFACE=true
     HF_MODEL_REPO=elsaelisa09/smartface-attendance-model
     MODEL_PATH=best_gacor.pth
     IMG_SIZE=224
     FRONTEND_URL=https://your-app.vercel.app
     PORT=5000
     HOST=0.0.0.0
     ```
   - **Note:** `FRONTEND_URL` akan diupdate setelah deploy frontend

6. **Deploy:**
   - Railway akan otomatis detect Python dan install dependencies
   - **First deploy:** Model akan didownload dari Hugging Face (~30-60 detik)
   - Build time total: ~5-10 menit (PyTorch + Model download)
   - Note URL backend: `https://your-backend.up.railway.app`
   - ✅ Model tersimpan di cache, restart selanjutnya cepat!

### Step 2B: Deploy ke Render (100% FREE! ✅ Recommended)

1. **Login ke Render:** https://render.com
2. **Create New Web Service** → Connect Repository
3. **Select repository:** `TugasBesarIIDeepLearning`
4. **Root Directory:** `SmartFace/backend`
5. **Settings:**
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Instance Type:** Free
6. **Environment Variables:**
   ```
   FLASK_ENV=production
   USE_HUGGINGFACE=true
   HF_MODEL_REPO=elsaelisa09/smartface-attendance-model
   MODEL_PATH=best_gacor.pth
   IMG_SIZE=224
   FRONTEND_URL=https://your-app.vercel.app
   ```
7. **Deploy:**
   - Build time: ~10-15 menit (max limit)
   - Model download otomatis dari Hugging Face
   - ⚠️ Free tier: Cold start 30-60 detik setelah 15 menit idle
   - ✅ 100% Gratis forever!

---

## 🌐 Part 2: Frontend Deployment (Vercel)

### Step 1: Setup Environment Variable

1. **Copy `.env.example` ke `.env`:**
```powershell
cd SmartFace
Copy-Item .env.example .env
```

2. **Edit `.env` file:**
```env
VITE_API_URL=https://your-backend.up.railway.app
```
*Ganti dengan URL backend dari Railway/Render*

### Step 2: Deploy ke Vercel

#### Option A: Via Vercel Dashboard (Easiest)

1. **Login ke Vercel:** https://vercel.com
2. **New Project** → Import Git Repository
3. **Select:** `elsaelisa09/TugasBesarIIDeepLearning`
4. **Root Directory:** Pilih `SmartFace` (folder dengan package.json)
5. **Framework Preset:** Vite
6. **Build Settings:**
   - Build Command: `npm run build` atau `bun run build`
   - Output Directory: `dist`
   - Install Command: Auto-detect
7. **Environment Variables:**
   - Key: `VITE_API_URL`
   - Value: `https://your-backend.up.railway.app`
8. **Deploy** - selesai dalam 1-2 menit
9. **Copy URL frontend:** `https://your-app.vercel.app`

#### Option B: Via Vercel CLI

```powershell
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
cd SmartFace
vercel --prod

# Follow prompts, set environment variable when asked
```

### Step 3: Update Backend CORS

1. **Kembali ke Railway/Render dashboard**
2. **Update environment variable:**
   ```
   FRONTEND_URL=https://your-app.vercel.app
   ```
3. **Redeploy backend** (Railway auto-redeploy on env change)

---

## ✅ Part 3: Verification

### Test Deployment

1. **Buka frontend URL:** `https://your-app.vercel.app`
2. **Test fitur:**
   - ✅ Kamera bisa aktif (HTTPS required - Vercel auto-provides)
   - ✅ Upload foto
   - ✅ Face detection
   - ✅ Attendance marking
   - ✅ History

### Check Backend Logs

**Railway:**
- Dashboard → Your service → Logs tab
- Cari: "Model: ✓ Loaded"
- Cari: "Face Detector: ✓ MTCNN"

**Render:**
- Dashboard → Your service → Logs

### Common Issues & Solutions

#### ❌ CORS Error
```
Access to fetch has been blocked by CORS policy
```
**Solution:** Update `FRONTEND_URL` di backend env variables

#### ❌ Model Not Loading
```
Error loading model
```
**Solution:** 
- Check logs: Cari pesan "📥 Downloading model from Hugging Face"
- Pastikan `USE_HUGGINGFACE=true` di environment variables
- Verify Hugging Face model: https://huggingface.co/elsaelisa09/smartface-attendance-model
- Check internet connection dari server (firewall blocked?)
- Fallback: Set `USE_HUGGINGFACE=false` dan commit model ke Git

#### ❌ Out of Memory (Railway/Render)
```
MemoryError or Killed
```
**Solution:**
- Railway: Upgrade plan ($5/month for 1GB RAM)
- Render: Free tier hanya 512MB, pertimbangkan paid tier
- Optimize model: Reduce batch size atau workers

#### ❌ Backend Cold Start (Render Free)
```
Backend takes 30-60s to respond
```
**Solution:**
- Render free tier sleeps after 15 min inactivity
- Upgrade to paid tier ($7/month) atau gunakan Railway

---

## 🔧 Part 4: Configuration Files Summary

### Backend Files Created:
- ✅ `backend/.env.example` - Template environment variables
- ✅ `backend/Procfile` - Railway/Render start command
- ✅ `backend/railway.json` - Railway configuration
- ✅ `backend/render.yaml` - Render configuration
- ✅ `backend/.gitignore` - Ignore sensitive files
- ✅ `backend/requirements.txt` - Updated with facenet-pytorch, gunicorn

### Frontend Files Created:
- ✅ `.env.example` - Template for API URL
- ✅ Updated `CameraView.tsx` - Use env variable
- ✅ Updated `Index.tsx` - Use env variable

---

## 📊 Cost Estimate

### 🆓 100% FREE (Recommended untuk Tugas Kuliah):
- **Vercel:** Free (Frontend) ✅
- **Render:** Free (Backend) ✅
- **Hugging Face:** Free (Model Storage) ✅
- **Total:** $0/month ✅
- ⚠️ Cold start 30-60 detik setelah idle

### Railway Trial:
- **Vercel:** Free (Frontend) ✅
- **Railway:** $5 trial credit (Backend - 1-2 minggu) ✅
- **Total:** FREE untuk testing

### Paid (Production):
- **Vercel:** Free (Frontend)
- **Railway:** $5/month (Backend - 1GB RAM, no cold starts)
- **Total:** $5/month ✅ Untuk production yang stabil

---

## 🔐 Security Notes

1. **Never commit `.env` file** - sudah ada di `.gitignore`
2. **Hugging Face Token:**
   - ⚠️ **NEVER share token in chat or commit to Git!**
   - Model public, tidak perlu token untuk download
   - Jika model private: Set `HF_TOKEN` di environment variables (bukan di code!)
3. **Model file (`best_gacor.pth`):**
   - ✅ Stored on Hugging Face (tidak di Git repository)
   - ✅ Auto-downloaded saat deployment
   - ✅ Public access (anyone can use)
4. **Attendance data:**
   - `attendance.json` auto-ignored via `.gitignore`
   - Consider adding database (PostgreSQL) for production

---

## 🚀 Quick Deploy Commands

### Initial Deployment:
```powershell
# 1. Commit all changes
git add .
git commit -m "Ready for deployment"
git push origin elsaaaa

# 2. Deploy frontend (Vercel CLI)
cd SmartFace
vercel --prod

# 3. Deploy backend (Railway/Render via dashboard)
# Follow steps in Part 1 above
```

### Update Deployment:
```powershell
# Make changes, then:
git add .
git commit -m "Update feature X"
git push origin elsaaaa

# Vercel & Railway will auto-deploy on git push ✅
```

---

## 📞 Support

Kalau ada masalah:
1. Check logs di Railway/Render dashboard
2. Check browser console (F12) untuk frontend errors
3. Test backend health: `https://your-backend.up.railway.app/health`

---

## ✨ Next Steps (Optional)

Setelah deployment sukses, pertimbangkan:

1. **Custom Domain:**
   - Vercel: Add custom domain (free HTTPS)
   - Railway: Add custom domain ($0.10/month)

2. **Database:**
   - Ganti `attendance.json` dengan PostgreSQL
   - Railway provides free PostgreSQL addon

3. **Model Management:**
   - ✅ Model sudah di Hugging Face Hub
   - Update model: Upload new version dengan `huggingface-cli upload`
   - Private model: Set repo private dan tambahkan `HF_TOKEN` env var

4. **Monitoring:**
   - Setup Sentry untuk error tracking
   - Add analytics (Vercel Analytics)

---

**🎉 Selamat! Aplikasi kamu sekarang live di internet!**

Frontend: `https://your-app.vercel.app`  
Backend: `https://your-backend.up.railway.app`
