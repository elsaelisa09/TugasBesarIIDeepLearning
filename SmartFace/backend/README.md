---
title: SmartFace Backend
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SmartFace Attendance Backend

Face recognition backend for SmartFace attendance system using ResNet50 with ArcFace.

## Features
- Face detection using MTCNN
- Face recognition using ResNet50 + ArcFace
- Attendance tracking
- CORS enabled for frontend integration

## API Endpoints
- `GET /` - API info
- `GET /health` - Health check
- `POST /recognize` - Face recognition
- `POST /mark-attendance` - Mark attendance
- `GET /attendance` - Get attendance records
- `DELETE /attendance/<id>` - Delete attendance record
