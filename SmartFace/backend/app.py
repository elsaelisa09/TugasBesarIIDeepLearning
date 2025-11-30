"""
SmartFace - Sistem Absensi Berbasis Face Recognition
Backend Flask Application dengan Integrated Frontend & API
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os
from datetime import datetime
import json

# ============================================================================
# FLASK APP SETUP
# ============================================================================
app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# Load Face Detection (MTCNN from facenet-pytorch)
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)
    print("✓ MTCNN (Facenet-PyTorch) loaded successfully")
    face_detector = 'mtcnn'
except Exception as e:
    print(f"⚠ MTCNN not available: {e}")
    mtcnn = None
    face_detector = None

# Device
device = torch.device('cpu')

# Model Definition (sama dengan training)
class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes=70):
        super(FineTunedResNet50, self).__init__()
        resnet = models.resnet50(pretrained=False)
        
        for param in resnet.parameters():
            param.requires_grad = False
        
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.model = resnet
        
    def forward(self, x):
        return self.model(x)

# Load Model
MODEL_PATH = 'best_finetuned_resnet50.pth'  # Model sekarang di folder backend
model = None
label_encoder = None
num_classes = 70

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    num_classes = len(label_encoder.classes_)
    
    model = FineTunedResNet50(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Classes: {num_classes}")
    print(f"  Labels: {label_encoder.classes_[:5]}... (showing first 5)")
except Exception as e:
    print(f"✗ Error loading model: {e}")

# Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Attendance storage
ATTENDANCE_FILE = 'attendance.json'

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_attendance(attendance_list):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(attendance_list, f, indent=2)

def parse_student_data():
    """Parse CSV to get student NIM and Kelas"""
    student_data = {}
    csv_path = '../labels-nim.csv'
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        nim = parts[1].strip()
                        kelas = parts[2].strip()
                        student_data[name] = {'nim': nim, 'kelas': kelas}
        except:
            pass
    
    return student_data

STUDENT_DATA = parse_student_data()

def save_attendance(attendance_list):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(attendance_list, f, indent=2)

def detect_and_crop_face(image_array):
    """Detect face using MTCNN (Facenet-PyTorch) and return cropped face"""
    if mtcnn is None:
        # Fallback: return center crop if MTCNN not available
        h, w = image_array.shape[:2]
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return image_array[y1:y1+size, x1:x1+size], None
    
    # Convert BGR to RGB for MTCNN
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # Detect faces with MTCNN
    boxes, probs = mtcnn.detect(image_pil)
    
    if boxes is None or len(boxes) == 0:
        return None, None
    
    # Get first detected face (highest confidence)
    box = boxes[0]
    x1, y1, x2, y2 = box.astype(int)
    
    # Safety clamp
    h, w = image_array.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Crop face
    face_crop = image_array[y1:y2, x1:x2]
    
    # Return crop and bbox
    bbox = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
    
    return face_crop, bbox

def predict_identity(face_image):
    """Predict identity from face image"""
    if model is None or label_encoder is None:
        return []
    
    # Convert to PIL
    if isinstance(face_image, np.ndarray):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
    
    # Transform
    img_tensor = test_transform(face_image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    predictions = []
    for prob, idx in zip(top3_prob, top3_idx):
        label = label_encoder.inverse_transform([idx.item()])[0]
        confidence = prob.item() * 100
        predictions.append({
            'label': label,
            'confidence': round(confidence, 2)
        })
    
    return predictions

# ============================================================================
# ROUTES - FRONTEND PAGES
# ============================================================================
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page with metrics"""
    return render_template('dashboard.html')

@app.route('/attendance-list')
def attendance_list_page():
    """Attendance records page"""
    return render_template('attendance.html')

# ============================================================================
# ROUTES - API ENDPOINTS
# ============================================================================
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'face_detector': face_detector,
        'mtcnn_loaded': mtcnn is not None,
        'num_classes': num_classes
    })

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        
        # Get image from base64
        image_data = data.get('image', '')
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Detect and crop face
        face_crop, bbox = detect_and_crop_face(image)
        
        if face_crop is None:
            return jsonify({'error': 'No face detected'}), 400
        
        # Predict identity
        predictions = predict_identity(face_crop)
        
        if not predictions:
            return jsonify({'error': 'Model not available'}), 500
        
        # Get student info
        top_student = predictions[0]['label']
        student_info = STUDENT_DATA.get(top_student, {'nim': 'N/A', 'kelas': 'N/A'})
        
        _, buffer = cv2.imencode('.jpg', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Draw bbox
        if bbox:
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 4)
        
        _, buffer = cv2.imencode('.jpg', image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'bbox': bbox,
            'face_image': f'data:image/jpeg;base64,{face_base64}',
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'predictions': predictions,
            'student_info': student_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.get_json()
        
        label = data.get('label')
        confidence = data.get('confidence')
        image = data.get('image')
        
        if not label or confidence is None:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get student info
        student_info = STUDENT_DATA.get(label, {'nim': 'N/A', 'kelas': 'N/A'})
        
        # Create attendance record
        attendance_record = {
            'id': len(load_attendance()) + 1,
            'label': label,
            'nim': student_info.get('nim', 'N/A'),
            'kelas': student_info.get('kelas', 'N/A'),
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'status': 'present'
        }
        
        # Load existing attendance
        attendance_list = load_attendance()
        
        # Check if already marked today
        today = datetime.now().strftime('%Y-%m-%d')
        already_marked = any(
            record['label'] == label and record['date'] == today 
            for record in attendance_list
        )
        
        if already_marked:
            return jsonify({
                'success': False,
                'message': f'{label} sudah absen hari ini'
            }), 400
        
        # Add and save
        attendance_list.append(attendance_record)
        save_attendance(attendance_list)
        
        return jsonify({
            'success': True,
            'message': f'Absensi {label} berhasil dicatat',
            'record': attendance_record
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/attendance', methods=['GET'])
def get_attendance():
    try:
        attendance_list = load_attendance()
        
        # Filter by date if provided
        date_filter = request.args.get('date')
        if date_filter:
            attendance_list = [
                record for record in attendance_list 
                if record['date'] == date_filter
            ]
        
        return jsonify({
            'success': True,
            'data': attendance_list,
            'total': len(attendance_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/attendance/<int:id>', methods=['DELETE'])
def delete_attendance(id):
    try:
        attendance_list = load_attendance()
        attendance_list = [record for record in attendance_list if record['id'] != id]
        save_attendance(attendance_list)
        
        return jsonify({
            'success': True,
            'message': 'Attendance record deleted'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get attendance statistics"""
    try:
        attendance_list = load_attendance()
        today = datetime.now().strftime('%Y-%m-%d')
        
        today_attendance = [r for r in attendance_list if r['date'] == today]
        unique_students = len(set(r['label'] for r in today_attendance))
        
        return jsonify({
            'success': True,
            'total_records': len(attendance_list),
            'today_attendance': len(today_attendance),
            'unique_students_today': unique_students,
            'model_accuracy': 98.5
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def print_banner():
    """Print startup banner"""
    banner = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║        🎯 SMARTFACE - FACE RECOGNITION ATTENDANCE SYSTEM v3.0              ║
║              Integrated Backend + Frontend Single Application               ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 SYSTEM INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    print(banner)
    
    model_status = "✅ READY" if model else "❌ FAILED"
    print(f"  Model Status       : {model_status}")
    print(f"  Model Type         : ResNet50 (Fine-tuned)")
    print(f"  Classes Recognized : {num_classes} students")
    print(f"  Model Accuracy     : 98.5%")
    
    detector_status = "✅ ACTIVE (MTCNN)" if mtcnn else "⚠️  FALLBACK (Center Crop)"
    print(f"\n  Face Detection     : {detector_status}")
    print(f"  Device             : CPU")
    
    print(f"\n  API Framework      : Flask 3.1.0")
    print(f"  CORS Enabled       : ✅ Yes")
    print(f"  Attendance DB      : JSON-based (Local)")
    
    print(f"\n" + "━"*80)
    print(f"\n🌐 API ENDPOINTS")
    print(f"━"*80)
    print(f"  GET    /api/health              → Check system status")
    print(f"  POST   /recognize               → Detect & recognize face")
    print(f"  POST   /mark-attendance         → Record attendance")
    print(f"  GET    /attendance              → Retrieve attendance records")
    print(f"  DELETE /attendance/<id>         → Delete attendance record")
    print(f"  GET    /api/stats               → Get statistics")
    
    print(f"\n" + "━"*80)
    print(f"\n📱 FRONTEND PAGES")
    print(f"━"*80)
    print(f"  GET    /                        → Main Dashboard")
    print(f"  GET    /dashboard               → Statistics & Metrics")
    print(f"  GET    /attendance-list         → View Records")
    
    print(f"\n" + "━"*80)
    print(f"\n🔧 CONFIGURATION")
    print(f"━"*80)
    print(f"  Server Address    : 0.0.0.0")
    print(f"  Server Port       : 5000")
    print(f"  Environment       : Development (Debug Mode)")
    print(f"  Model Path        : best_finetuned_resnet50.pth")
    print(f"  Attendance File   : attendance.json")
    
    print(f"\n" + "━"*80)
    print(f"\n📝 FEATURES")
    print(f"━"*80)
    print(f"  ✨ Real-time face detection and recognition")
    print(f"  ✨ Support for 70 student identities")
    print(f"  ✨ Automatic attendance recording with timestamp")
    print(f"  ✨ Duplicate detection (1 student = 1 attendance/day)")
    print(f"  ✨ RESTful API for integration")
    print(f"  ✨ Integrated web-based frontend")
    print(f"  ✨ Real-time dashboard with statistics")
    print(f"  ✨ Attendance records management")
    
    print(f"\n" + "="*80)
    print(f"\n⏳ System initialization: COMPLETE")
    print(f"🚀 Server ready to accept requests!")
    
    print(f"\n💡 Access the application at:")
    print(f"   Frontend: http://127.0.0.1:5000/")
    print(f"   API Docs: http://127.0.0.1:5000/api/health")
    
    print(f"\n" + "="*80 + "\n")

if __name__ == '__main__':
    print_banner()
    
    # Validate model before starting
    if model is None:
        print("⚠️  WARNING: Model not loaded!")
        print("   Please check if 'best_finetuned_resnet50.pth' exists in backend folder")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
