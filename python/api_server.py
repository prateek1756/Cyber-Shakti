from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from deepfake_detector import DeepfakeDetector
import json
from datetime import datetime

# #region agent log
def debug_log(location, message, data, hypothesis_id):
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open(r"c:\projects\CyberShakti\.cursor\debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass
# #endregion

app = Flask(__name__)
CORS(app)

# Initialize detector
# #region agent log
debug_log("api_server.py:12", "Before DeepfakeDetector initialization", {}, "A")
# #endregion
try:
    detector = DeepfakeDetector()
    # #region agent log
    debug_log("api_server.py:13", "After DeepfakeDetector initialization", {"detector_created": True}, "A")
    # #endregion
    detector.load_training_data()
    # #region agent log
    debug_log("api_server.py:14", "After load_training_data", {"training_samples": len(detector.training_data)}, "A")
    # #endregion
except Exception as e:
    # #region agent log
    debug_log("api_server.py:13", "DeepfakeDetector initialization failed", {"error": str(e), "error_type": type(e).__name__}, "A")
    # #endregion
    raise

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/api/deepfake/analyze', methods=['POST'])
def analyze_deepfake():
    """Analyze uploaded image/video for deepfake detection"""
    # #region agent log
    debug_log("api_server.py:20", "analyze_deepfake entry", {"method": request.method}, "D")
    # #endregion
    try:
        # #region agent log
        debug_log("api_server.py:24", "Checking request.files", {"has_files": 'file' in request.files}, "D")
        # #endregion
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        # #region agent log
        debug_log("api_server.py:27", "File received", {"filename": file.filename, "content_type": file.content_type}, "D")
        # #endregion
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{datetime.now().timestamp()}_{file.filename}")
        # #region agent log
        debug_log("api_server.py:32", "Before file.save", {"temp_path": temp_path}, "D")
        # #endregion
        file.save(temp_path)
        # #region agent log
        debug_log("api_server.py:33", "After file.save", {"file_exists": os.path.exists(temp_path)}, "D")
        # #endregion
        
        try:
            # Analyze the file
            # #region agent log
            debug_log("api_server.py:36", "Before detect_deepfake", {"temp_path": temp_path}, "D")
            # #endregion
            result = detector.detect_deepfake(temp_path)
            # #region agent log
            debug_log("api_server.py:37", "After detect_deepfake", {"result_keys": list(result.keys()) if result else None}, "D")
            # #endregion
            
            # Add metadata
            file_stats = os.stat(temp_path)
            result['metadata'] = {
                'filename': file.filename,
                'file_size': file_stats.st_size,
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(result)
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        # #region agent log
        debug_log("api_server.py:108", "analyze_deepfake exception", {"error": str(e), "error_type": type(e).__name__}, "D")
        # #endregion
        return jsonify({'error': str(e)}), 500

@app.route('/api/deepfake/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for model training"""
    # #region agent log
    debug_log("api_server.py:57", "submit_feedback entry", {"method": request.method, "content_type": request.content_type}, "E")
    # #endregion
    try:
        # #region agent log
        debug_log("api_server.py:61", "Before request.get_json", {"has_files": 'file' in request.files}, "E")
        # #endregion
        data = request.get_json()
        # #region agent log
        debug_log("api_server.py:61", "After request.get_json", {"data": data, "data_type": type(data).__name__}, "E")
        # #endregion
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        # #region agent log
        debug_log("api_server.py:66", "File from request.files", {"filename": file.filename}, "E")
        # #endregion
        # Try to get is_deepfake from form data if JSON is None
        if data is None:
            is_deepfake_str = request.form.get('is_deepfake', 'false')
            # #region agent log
            debug_log("api_server.py:137", "Getting is_deepfake from form", {"is_deepfake_str": is_deepfake_str}, "E")
            # #endregion
            # Parse JSON string if it's JSON, otherwise treat as boolean string
            try:
                is_deepfake = json.loads(is_deepfake_str)
            except (json.JSONDecodeError, TypeError):
                is_deepfake = is_deepfake_str.lower() in ('true', '1')
        else:
            is_deepfake = data.get('is_deepfake', False)
        # #region agent log
        debug_log("api_server.py:67", "is_deepfake value", {"is_deepfake": is_deepfake}, "E")
        # #endregion
        
        # Save file for training
        training_path = os.path.join(UPLOAD_DIR, f"training_{datetime.now().timestamp()}_{file.filename}")
        file.save(training_path)
        
        # Add to training data
        success = detector.add_training_sample(training_path, is_deepfake, retrain=True)
        
        if success:
            return jsonify({'message': 'Feedback submitted successfully'})
        else:
            return jsonify({'error': 'Could not process feedback'}), 400
    
    except Exception as e:
        # #region agent log
        debug_log("api_server.py:157", "submit_feedback exception", {"error": str(e), "error_type": type(e).__name__}, "E")
        # #endregion
        return jsonify({'error': str(e)}), 500

@app.route('/api/deepfake/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    # #region agent log
    debug_log("api_server.py:84", "get_stats entry", {}, "F")
    # #endregion
    try:
        stats = {
            'training_samples': len(detector.training_data),
            'model_loaded': detector.model is not None,
            'last_updated': datetime.now().isoformat()
        }
        # #region agent log
        debug_log("api_server.py:93", "get_stats success", {"stats": stats}, "F")
        # #endregion
        return jsonify(stats)
    
    except Exception as e:
        # #region agent log
        debug_log("api_server.py:95", "get_stats exception", {"error": str(e), "error_type": type(e).__name__}, "F")
        # #endregion
        return jsonify({'error': str(e)}), 500

@app.route('/api/deepfake/retrain', methods=['POST'])
def retrain_model():
    """Manually trigger model retraining"""
    try:
        success = detector.retrain_model()
        if success:
            return jsonify({'message': 'Model retrained successfully'})
        else:
            return jsonify({'error': 'Not enough training data (minimum 10 samples)'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # #region agent log
    debug_log("api_server.py:111", "API server starting", {}, "A")
    # #endregion
    print("Starting Deepfake Detection API...")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    app.run(host='0.0.0.0', port=5001, debug=True)