from flask import Flask, render_template, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask application initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variable for model
model = None

def load_model():
    """Load the trained model with error handling"""
    global model
    try:
        model_path = 'models/best_cnn_model.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def preprocess_image(image_file):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Open and convert image
        img = Image.open(BytesIO(image_file.read()))
        
        # Convert to RGB if necessary (handles different image modes)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model's expected input size
        img = img.resize((128, 128))
        
        # Convert to array and normalize
        img_array = img_to_array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e

def validate_image_file(file):
    """Validate uploaded file"""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        return False, "Invalid file format. Please upload PNG, JPG, JPEG, BMP, or TIFF files only"
    
    return True, "Valid file"

# Routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please contact administrator.'
            }), 500
        
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided in request'
            }), 400
        
        file = request.files['file']
        
        # Validate file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
        # Preprocess image
        try:
            img_array = preprocess_image(file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 400
        
        # Make prediction
        try:
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            
            # Class names (make sure this matches your model's training)
            class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
            predicted_class = class_names[predicted_class_index]
            
            # Get all class probabilities for detailed results
            class_probabilities = {}
            for i, class_name in enumerate(class_names):
                class_probabilities[class_name] = float(prediction[0][i])
            
            logger.info(f"Prediction successful: {predicted_class} (confidence: {confidence:.2f})")
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': class_probabilities
            })
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error during prediction: {str(e)}'
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        model_status = "loaded" if model is not None else "not loaded"
        return jsonify({
            'status': 'healthy',
            'model_status': model_status
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size allowed is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }), 500

# Initialize the application
def create_app():
    """Application factory"""
    try:
        # Load the model
        load_model()
        logger.info("Flask application initialized successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # Create the application
        app = create_app()
        
        # Run the application
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")
        print("Please ensure:")
        print("1. The model file exists at 'models/best_cnn_model.keras'")
        print("2. All required packages are installed")
        print("3. The templates folder exists with index.html")