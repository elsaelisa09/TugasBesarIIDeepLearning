from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import base64
import io
import pickle
import os
from datetime import datetime
import json
import torch.nn.functional as F

app = Flask(__name__)
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
class ResNet50Embedding(nn.Module):
    def __init__(self, embed_dim=512, p_drop=0.5):
        super(ResNet50Embedding, self).__init__()
        resnet = models.resnet50(weights=None)  # atau pretrained=False

        # Ambil fitur sebelum FC
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.backbone = resnet
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim) 

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x) 
        return x

# Load ArcFace checkpoint
MODEL_PATH = 'resnet50_arcface_best2.pt'
model = None
arc_weight = None
idx_to_class_map = {}
num_classes = 0
IMG_SIZE = 224

try:
    ckpt = torch.load(MODEL_PATH, map_location=device)

    # --- Ambil info kelas ---
    class_to_idx = ckpt.get("class_to_idx", {})
    idx_to_class = ckpt.get("idx_to_class", {})
    num_classes = len(class_to_idx) if class_to_idx else 70

    # Normalisasi idx_to_class → dict idx:int -> label:str
    if isinstance(idx_to_class, list):
        idx_to_class_map = {i: lbl for i, lbl in enumerate(idx_to_class)}
    elif isinstance(idx_to_class, dict) and all(isinstance(k, int) for k in idx_to_class.keys()):
        idx_to_class_map = idx_to_class
    elif isinstance(idx_to_class, dict) and all(isinstance(v, int) for v in idx_to_class.values()):
        idx_to_class_map = {v: k for k, v in idx_to_class.items()}
    else:
        # fallback kalau gagal, bikin label generik
        idx_to_class_map = {i: f"class_{i}" for i in range(num_classes)}

    # --- ukuran gambar dari ckpt (kalau ada) ---
    IMG_SIZE = ckpt.get("img_size", 224)

    # --- Bangun model embedding dan load state_dict ---
    model = ResNet50Embedding(embed_dim=512, p_drop=0.5)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    # Optional: print info kalau mau lihat apa yang di-skip
    missing_keys, unexpected_keys = load_result
    if unexpected_keys:
        print("  [WARN] Unexpected keys ignored in model state_dict:", unexpected_keys)
    if missing_keys:
        print("  [WARN] Missing keys in model state_dict:", missing_keys)

    model.to(device).eval()

    # --- Ambil weight ArcFace ---
    arc_state = ckpt["arc"]
    # kalau yg disimpan adalah state_dict:
    if isinstance(arc_state, dict) and "weight" in arc_state:
        arc_weight = arc_state["weight"]
    else:
        # kalau yg disimpan module, ambil atribut weight
        arc_weight = arc_state.weight
    arc_weight = arc_weight.to(device)

    print("✓ ArcFace checkpoint loaded successfully!")
    print(f"  Classes: {num_classes}")
    print(f"  Sample labels: {[idx_to_class_map[i] for i in list(idx_to_class_map.keys())[:5]]} ...")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    print(f"  Make sure {MODEL_PATH} exists in the backend folder")

# Transform
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
    """Predict identity from face image (ArcFace ResNet50)"""
    if model is None or arc_weight is None or not idx_to_class_map:
        return []

    # Convert to PIL
    if isinstance(face_image, np.ndarray):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)

    # Transform
    img_tensor = test_transform(face_image).unsqueeze(0).to(device)

    # Predict: embedding -> cosine similarity with arc_weight
    with torch.no_grad():
        emb = model(img_tensor)                           # [1, 512]
        emb_norm = F.normalize(emb, dim=1)                # [1, 512]
        w_norm = F.normalize(arc_weight, dim=1)           # [C, 512]
        logits = torch.matmul(emb_norm, w_norm.t())       # [1, C]
        probabilities = torch.softmax(logits, dim=1)[0]   # [C]

    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)

    predictions = []
    for prob, idx in zip(top3_prob, top3_idx):
        idx_int = idx.item()
        label = idx_to_class_map.get(idx_int, f"class_{idx_int}")
        confidence = prob.item() * 100
        predictions.append({
            'label': label,
            'confidence': round(confidence, 2)
        })

    return predictions

@app.route('/health', methods=['GET'])
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
        
        # Encode cropped face to base64
        _, buffer = cv2.imencode('.jpg', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Draw bbox on original image
        if bbox:
            # Convert to PIL Image for custom font
            from PIL import ImageDraw, ImageFont
            
            # Draw rectangle with cv2
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 4)
            
            # Convert to PIL for text with custom font
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            
            # Try to load Poppins font, fallback to default if not available
            label_text = predictions[0]['label']
            font_size = 40
            try:
                # Try to load Poppins Bold
                font = ImageFont.truetype("C:/Windows/Fonts/Poppins-Bold.ttf", font_size)
            except:
                try:
                    # Try alternative font paths or fallback
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    # Use default font
                    font = ImageFont.load_default()
            
            # Calculate text size and center position
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            text_x = bbox['x1'] + (bbox['x2'] - bbox['x1'] - text_width) // 2
            text_y = bbox['y1'] - text_height - 30
            
            # Pastikan text tidak keluar dari frame
            if text_x < 5:
                text_x = 5
            if text_y < 5:
                text_y = bbox['y2'] + 10
            
            # Draw text with outline for better visibility
            outline_color = (0, 0, 0)
            text_color = (0, 255, 0)
            
            # Draw outline
            for adj_x in [-2, -1, 0, 1, 2]:
                for adj_y in [-2, -1, 0, 1, 2]:
                    draw.text((text_x + adj_x, text_y + adj_y), label_text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((text_x, text_y), label_text, font=font, fill=text_color)
            
            # Convert back to cv2 format
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Encode annotated image to base64
        _, buffer = cv2.imencode('.jpg', image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'bbox': bbox,
            'face_image': f'data:image/jpeg;base64,{face_base64}',
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'predictions': predictions
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
        
        # Create attendance record
        attendance_record = {
            'id': len(load_attendance()) + 1,
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'status': 'present',
            'image': image
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

if __name__ == '__main__':
    print("="*80)
    print("🚀 STARTING FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*80)
    print(f"Model: {'✓ Loaded' if model else '✗ Not loaded'}")
    print(f"Face Detector: {'✓ MTCNN' if mtcnn else '✗ Not loaded'}")
    print(f"Classes: {num_classes}")
    print("="*80)
    app.run(debug=True, host='0.0.0.0', port=5000)
