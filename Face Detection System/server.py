
from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging
import time

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_data):
    try:
        
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
       
        image_np = np.array(image)
        
       
        if len(image_np.shape) == 4:
            image_np = image_np[:, :, :3]
            
        return image_np
    except Exception as e:
        logger.error(f"छवि प्रीप्रोसेसिंग त्रुटि: {str(e)}")
        raise

@app.route('/analyze', methods=['POST'])
def analyze_face():
    try:
        start_time = time.time()
        
        
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'कोई छवि डेटा नहीं मिला'}), 400
            
        
        image_np = preprocess_image(image_data)
        
        # चेहरा विश्लेषण
        analysis = DeepFace.analyze(
            image_np,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # परिणाम तैयार करें
        result = {
            'emotion': analysis[0]['emotion'],
            'dominant_emotion': analysis[0]['dominant_emotion'],
            'age': analysis[0]['age'],
            'gender': analysis[0]['gender'],
            'processing_time': f"{(time.time() - start_time):.2f}s"
        }
        
        logger.info(f"विश्लेषण सफल: {result['processing_time']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"विश्लेषण त्रुटि: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)