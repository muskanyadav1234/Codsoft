from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import base64
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    static_url_path='', 
    static_folder='static',
    template_folder='templates')

# Initialize AI models - Using MobileNetV2 for faster processing
try:
    logger.info("Loading AI models...")
    image_model = tf.keras.applications.MobileNetV2(weights='imagenet')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Failed to load face cascade classifier")
    logger.info("AI models loaded successfully")
except Exception as e:
    logger.error(f"Error initializing AI models: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_image(image):
    try:
        logger.info("Preprocessing image...")
        # Optimize image size for faster processing
        target_size = (224, 224)  # Standard size for MobileNetV2
        image = image.resize(target_size, Image.LANCZOS)
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info("Image preprocessing complete")
        return image
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if not request.json or 'image' not in request.json:
        logger.warning("No image data received")
        return jsonify({'error': 'No image data received'}), 400

    try:
        # Extract and decode image
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        logger.info("Decoding image data...")
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process image
        logger.info("Processing image...")
        image = preprocess_image(image)
        results = process_image(image)
        logger.info("Image processing complete")
        return jsonify(results)
    except base64.binascii.Error:
        logger.error("Invalid image data format")
        return jsonify({'error': 'Invalid image data format'}), 400
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_image(image):
    try:
        logger.info("Converting to OpenCV format...")
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        logger.info("Detecting faces...")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        logger.info(f"Found {len(faces)} faces")
        
        logger.info("Running image classification...")
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = image_model.predict(img_array, verbose=0)
        results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)[0]
        logger.info("Classification complete")
        
        # Format results
        classifications = []
        for _, label, score in results[:2]:  # Reduced to top 2 predictions
            label = label.replace('_', ' ').title()
            classifications.append({
                'label': label,
                'confidence': float(score)
            })
        
        return {
            'faces_detected': len(faces),
            'classifications': classifications,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        raise

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure the models are loaded before starting the server
    if image_model is None or face_cascade.empty():
        logger.error("AI models not properly initialized")
        exit(1)
    
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5000)