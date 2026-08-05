from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
from PIL import Image
import numpy as np
import os
import json
import logging
import datetime
import uuid

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- PDF report generation (reportlab) -------------------------------
# Only used by the new /generate_report route below. Nothing about the
# existing model loading, preprocessing, validation, or prediction logic
# is touched by this addition.
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask application initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model + the class names it was trained with.
# class_names is loaded from models/class_names.json (written by
# train_model.py) instead of being hardcoded here, so the app can never
# drift out of sync with what the model actually learned to predict.
model = None
class_names = None

def load_model():
    """Load the trained TFLite model and its class-name mapping, with error handling."""
    global model, class_names
    try:
        model_path = os.path.join(BASE_DIR, 'best_cnn_model.tflite')
        class_names_path = os.path.join(BASE_DIR, 'models', 'class_names.json')

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Run train_model.py first."
            )
        if not os.path.exists(class_names_path):
            logger.error(f"Class names file not found at {class_names_path}")
            raise FileNotFoundError(
                f"Class names file not found at {class_names_path}. Run train_model.py first."
            )

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        with open(class_names_path, 'r') as f:
            class_names = json.load(f)

        output_details = interpreter.get_output_details()
        if output_details:
            output_shape = output_details[0]['shape']
            output_units = output_shape[-1] if len(output_shape) > 0 else None
        else:
            output_units = None

        if output_units is not None and output_units != len(class_names):
            raise ValueError(
                f"Model output has {output_units} units but class_names.json has "
                f"{len(class_names)} entries ({class_names}). The model and the "
                "class-name mapping are out of sync -- retrain with train_model.py."
            )

        model = {
            'interpreter': interpreter,
            'input_details': interpreter.get_input_details(),
            'output_details': output_details,
        }

        logger.info(f"Model loaded successfully. Classes: {class_names}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e


def get_model_bundle():
    """Load the model lazily on first use so Vercel cold starts stay lightweight."""
    global model, class_names
    if model is None:
        load_model()
    return model, class_names

def preprocess_image(image_file):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Open and convert image
        original_img = Image.open(BytesIO(image_file.read()))

        # Convert to RGB if necessary (handles different image modes)
        if original_img.mode != 'RGB':
            original_img = original_img.convert('RGB')

        # BUGFIX: validate_retinal_image() used to be called on the
        # 112x112 model-input copy below, which is downscaled enough to
        # wash out the edge/texture detail the validator relies on --
        # real fundus photos were being falsely rejected as "not a
        # retinal image" purely because of the resize, not because of
        # anything wrong with the photo. Keep the original-resolution
        # image around so validation runs on it instead.
        img = original_img.resize((112, 112))

        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, original_img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e


def validate_retinal_image(image):
    """Lightweight retinal-image validation before disease prediction."""
    try:
        if image is None:
            return False, "Invalid image. Please upload a valid retinal fundus image. Other image types, such as selfies, documents, screenshots, or non-eye images, are not supported."

        img_rgb = image.convert('RGB')
        img_array = np.array(img_rgb, dtype=np.float32)
        height, width = img_array.shape[:2]

        if width < 128 or height < 128:
            return False, "Invalid image. Please upload a valid retinal fundus image. Other image types, such as selfies, documents, screenshots, or non-eye images, are not supported."

        aspect_ratio = max(width, height) / max(1, min(width, height))
        if aspect_ratio > 2.5:
            return False, "Invalid image. Please upload a valid retinal fundus image. Other image types, such as selfies, documents, screenshots, or non-eye images, are not supported."

        gray = np.mean(img_array, axis=2)
        # BUGFIX: np.diff(gray, axis=0) has shape (H-1, W) and
        # np.diff(gray, axis=1) has shape (H, W-1) -- these cannot be added
        # directly (shape mismatch). The original code raised an exception
        # here on every single call, which silently rejected every
        # uploaded image (the except block below always returned False).
        # Crop both to the shared (H-1, W-1) region before combining.
        diff_v = np.abs(np.diff(gray, axis=0))[:, :-1]
        diff_h = np.abs(np.diff(gray, axis=1))[:-1, :]
        edge_diff = diff_v + diff_h
        edge_density = float(np.mean(edge_diff > 20))

        rg = np.abs(img_array[:, :, 0] - img_array[:, :, 1])
        yb = np.abs(0.5 * (img_array[:, :, 0] + img_array[:, :, 1]) - img_array[:, :, 2])
        colorfulness = float(np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2)))

        center_region = gray[int(height * 0.2):int(height * 0.8), int(width * 0.2):int(width * 0.8)]
        center_brightness = float(np.mean(center_region))
        overall_brightness = float(np.mean(gray))

        # Fundus-specific color check: retinal photos are consistently
        # warm-toned (red/orange from the vascular tissue) across almost
        # every non-background pixel -- this is true regardless of which
        # disease is present. Generic photos (landscapes, selfies,
        # documents, random objects) essentially never match this pattern,
        # since they mix in blues/greens/grays from sky, foliage, clothing,
        # backgrounds, etc. This check was added because colorfulness +
        # edge_density alone (the previous checks) only rule out flat/blank
        # images -- they don't verify the image actually looks like a
        # fundus photo, so ordinary photos were passing through and being
        # sent to the model instead of being rejected. Thresholds were
        # calibrated against this project's real training images (10th
        # percentile red-dominance was 0.72-0.99 and warmth 0.09-0.23
        # across all 4 classes) and validated against synthetic
        # non-fundus test images (landscapes, skin tones, documents,
        # posters) to confirm they're rejected.
        content_mask = gray >= 30  # exclude near-black background/corners
        if content_mask.sum() < 50:
            frac_red_dominant = 0.0
            warmth = 0.0
        else:
            r_content = img_array[:, :, 0][content_mask]
            g_content = img_array[:, :, 1][content_mask]
            b_content = img_array[:, :, 2][content_mask]
            frac_red_dominant = float(np.mean(r_content > b_content + 10))
            warmth = float(np.mean((r_content - b_content) / (r_content + g_content + b_content + 1e-6)))

        is_plausible_retinal = (
            colorfulness > 8.0 and
            edge_density > 0.003 and
            center_brightness < 220 and
            overall_brightness < 240 and
            overall_brightness > 25 and
            frac_red_dominant > 0.55 and
            warmth > 0.06
        )

        if not is_plausible_retinal:
            return False, "Invalid image. Please upload a valid retinal fundus image. Other image types, such as selfies, documents, screenshots, or non-eye images, are not supported."

        return True, "Valid retinal image"
    except Exception as e:
        logger.error(f"Retinal image validation failed: {str(e)}")
        return False, "Invalid image. Please upload a valid retinal fundus image. Other image types, such as selfies, documents, screenshots, or non-eye images, are not supported."


def validate_image_file(file):
    """Validate uploaded file"""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {
        '.jpg', '.jpeg', '.jfif', '.png', '.gif', '.webp', '.svg', '.bmp',
        '.tiff', '.tif', '.ico', '.avif', '.heic', '.heif'
    }
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        return False, "Invalid image. Please upload a valid retinal fundus image. Other image types, such as selfies, documents, screenshots, or non-eye images, are not supported."
    
    return True, "Valid file"

# Routes
# --- PDF report generation ------------------------------------------
# Short, plain-language descriptions shown in the PDF report. Falls back
# gracefully to a generic line for any class name not listed here, so
# this never breaks if the model is retrained with different classes.
CONDITION_DESCRIPTIONS = {
    'cataract': "Clouding of the eye's natural lens, which can cause blurred or dimmed vision. "
                "Commonly age-related and often treatable with surgery.",
    'diabetic_retinopathy': "Damage to the blood vessels of the retina associated with diabetes. "
                             "Early detection and blood sugar management are important to prevent vision loss.",
    'glaucoma': "A group of eye conditions that damage the optic nerve, often linked to elevated eye "
                "pressure. Early treatment can help slow or prevent further vision loss.",
    'normal': "No signs of cataract, diabetic retinopathy, or glaucoma were detected in this screening.",
}

REPORT_BRAND_COLOR = colors.HexColor('#2563eb')
REPORT_MUTED_COLOR = colors.HexColor('#64748b')


def generate_pdf_report(prediction, confidence, probabilities, image_bytes=None):
    """Build a PDF screening report from an already-computed prediction.

    This does not re-run the model or touch any prediction/validation
    logic -- it only formats results that were already produced by the
    existing /predict route into a downloadable PDF.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
        title="Retinal Screening Report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle', parent=styles['Title'], fontSize=20,
        textColor=colors.HexColor('#0f172a'), spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name='ReportSubtitle', parent=styles['Normal'], fontSize=10.5,
        textColor=REPORT_MUTED_COLOR, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name='SectionHeading', parent=styles['Heading2'], fontSize=13,
        textColor=colors.HexColor('#0f172a'), spaceBefore=16, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='ResultLabel', parent=styles['Normal'], fontSize=11,
        textColor=REPORT_MUTED_COLOR,
    ))
    styles.add(ParagraphStyle(
        name='ResultValue', parent=styles['Normal'], fontSize=22,
        textColor=REPORT_BRAND_COLOR, spaceAfter=4, leading=26,
    ))
    styles.add(ParagraphStyle(
        name='BodyText2', parent=styles['Normal'], fontSize=10, leading=14,
    ))
    styles.add(ParagraphStyle(
        name='Disclaimer', parent=styles['Normal'], fontSize=8.3, leading=11.5,
        textColor=REPORT_MUTED_COLOR,
    ))

    story = []

    # --- Header -----------------------------------------------------
    story.append(Paragraph("Retinal Fundus Screening Report", styles['ReportTitle']))
    story.append(Paragraph(
        "AI-assisted preliminary screening &mdash; generated by the Eye Disease Classification tool",
        styles['ReportSubtitle'],
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
    story.append(Spacer(1, 10))

    report_id = str(uuid.uuid4())[:8].upper()
    generated_at = datetime.datetime.now().strftime('%B %d, %Y at %H:%M')
    meta_table = Table(
        [["Report ID", report_id], ["Generated", generated_at]],
        colWidths=[35 * mm, 120 * mm],
    )
    meta_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), REPORT_MUTED_COLOR),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#0f172a')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # --- Uploaded image (if provided) + headline result side by side --
    result_label = str(prediction).replace('_', ' ').title()
    is_normal = str(prediction).lower() == 'normal'

    result_block = [
        Paragraph("SCREENING RESULT", styles['ResultLabel']),
        Paragraph(result_label, styles['ResultValue']),
        Paragraph(f"Confidence: {confidence * 100:.1f}%", styles['BodyText2']),
        Spacer(1, 6),
        Paragraph(
            CONDITION_DESCRIPTIONS.get(str(prediction).lower(),
                                        "No description available for this class."),
            styles['BodyText2'],
        ),
    ]

    if image_bytes:
        try:
            img_reader = BytesIO(image_bytes)
            pil_img = Image.open(img_reader).convert('RGB')
            pil_img.thumbnail((260, 260))
            thumb_buf = BytesIO()
            pil_img.save(thumb_buf, format='JPEG', quality=88)
            thumb_buf.seek(0)
            rl_img = RLImage(thumb_buf, width=55 * mm, height=55 * mm)
            row = Table([[rl_img, result_block]], colWidths=[60 * mm, 95 * mm])
            row.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (1, 0), (1, 0), 12),
            ]))
            story.append(row)
        except Exception as e:
            logger.warning(f"Could not embed image thumbnail in report: {e}")
            story.extend(result_block)
    else:
        story.extend(result_block)

    story.append(Spacer(1, 8))

    # --- Probability breakdown table ---------------------------------
    story.append(Paragraph("Detailed Probability Breakdown", styles['SectionHeading']))

    sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    table_data = [["Condition", "Probability"]]
    for cls, prob in sorted_probs:
        table_data.append([cls.replace('_', ' ').title(), f"{prob * 100:.1f}%"])

    prob_table = Table(table_data, colWidths=[110 * mm, 45 * mm])
    prob_table_style = [
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, 0), 0.75, colors.HexColor('#cbd5e1')),
        ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]
    # Highlight the row matching the predicted class
    for i, (cls, _prob) in enumerate(sorted_probs, start=1):
        if cls == prediction:
            prob_table_style.append(('TEXTCOLOR', (0, i), (-1, i), REPORT_BRAND_COLOR))
            prob_table_style.append(('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'))
    prob_table.setStyle(TableStyle(prob_table_style))
    story.append(prob_table)

    # --- Disclaimer ----------------------------------------------------
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Important Notice", styles['SectionHeading']))
    story.append(Paragraph(
        "This report is generated by an automated image-classification model and is intended "
        "for preliminary screening and informational purposes only. It is <b>not a medical "
        "diagnosis</b> and should not replace evaluation by a qualified ophthalmologist or "
        "other licensed eye-care professional. Model accuracy varies by condition -- see the "
        "project's classification report for details -- and any result, especially one "
        "suggesting a condition is present, should be confirmed with an in-person clinical "
        "examination. If you have concerns about your vision or eye health, please consult a "
        "healthcare provider promptly.",
        styles['Disclaimer'],
    ))

    def _footer(canvas_obj, _doc):
        canvas_obj.saveState()
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(REPORT_MUTED_COLOR)
        canvas_obj.drawString(18 * mm, 10 * mm, f"Report {report_id}")
        canvas_obj.drawRightString(
            A4[0] - 18 * mm, 10 * mm, "Eye Disease Classification -- AI Screening Tool"
        )
        canvas_obj.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buffer.seek(0)
    return buffer


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate a downloadable PDF report from an already-computed prediction.

    This route does NOT re-run the model or re-validate the image -- it
    only formats results the client already received from /predict into
    a PDF. The image file is optional and, if provided, is embedded as a
    thumbnail purely for the report's visual reference.
    """
    try:
        prediction = request.form.get('prediction')
        confidence_raw = request.form.get('confidence')
        probabilities_raw = request.form.get('probabilities')

        if not prediction or confidence_raw is None or not probabilities_raw:
            return jsonify({
                'success': False,
                'error': 'Missing prediction data. Please analyze an image first.'
            }), 400

        try:
            confidence = float(confidence_raw)
            probabilities = json.loads(probabilities_raw)
        except (ValueError, json.JSONDecodeError) as e:
            return jsonify({
                'success': False,
                'error': f'Invalid prediction data: {str(e)}'
            }), 400

        image_bytes = None
        if 'file' in request.files and request.files['file'].filename:
            image_bytes = request.files['file'].read()

        pdf_buffer = generate_pdf_report(prediction, confidence, probabilities, image_bytes)

        filename = f"retinal_screening_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error generating report: {str(e)}'
        }), 500


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests"""
    try:
        # Check if model is loaded
        model_bundle, class_names_for_prediction = get_model_bundle()
        if model_bundle is None:
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
            img_array, img = preprocess_image(file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }), 400

        is_retinal, validation_message = validate_retinal_image(img)
        if not is_retinal:
            return jsonify({
                'success': False,
                'error': validation_message
            }), 400
        
        # Make prediction
        try:
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))

            # class_names is loaded from models/class_names.json when the
            # model is first needed, so it always matches exactly what the
            # model was trained on -- no more hardcoded, drifted label lists.
            predicted_class = class_names_for_prediction[predicted_class_index]

            # Get all class probabilities for detailed results
            class_probabilities = {}
            for i, class_name in enumerate(class_names_for_prediction):
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
            'model_status': model_status,
            'classes': class_names if class_names is not None else []
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
        logger.info("Flask application initialized successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise e

# Create the app without eagerly loading the model so serverless cold starts stay lightweight.
app = create_app()

if __name__ == "__main__":
    try:
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