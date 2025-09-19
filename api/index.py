from flask import Flask, request, jsonify
import requests
import json
import base64
from werkzeug.utils import secure_filename
import os
from io import BytesIO
from PIL import Image
import tempfile

app = Flask(__name__)

# Configuration
API_KEY = "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"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# OCR Prompts
PROMPTS = {
    "DRIVER_LICENCE": """
just read form the image Perform OCR on the uploaded arabic driving licence Extract the following fields from this driving licence (and insurance if present) and return JSON:
                                    - Name(arabic)
                                    - First name(arabic)  
                                    - Date of birth  
                                    - Address
                                    - Country  
                                    - National identification number 
                                    - Driving licence number  
                                    - Groups (A, B…)  
                                    - Valid until  
                                    - Insurance company
""",
    "CAR_PLATE": """Perform OCR on the image.  
If a car plate is visible, extract and return only the license plate number as plain text.  
If no plate is found, return: "".
""",
    "CARTE_GRIS": """
extract from image Carte Gris algeria this information i need correct answer: 
  plate
  make
  model
  vin 
  first_registration
  category
  fiscal_power_cv
  ptac_kg
  color
  certificate_number
  owner_name
  owner_address
"""
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_from_bytes(image_bytes):
    """Encode image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')


def process_ocr(base64_image, ocr_type):
    """Process OCR request using OpenRouter API"""
    try:
        if ocr_type not in PROMPTS:
            return {"error": f"Invalid OCR type. Allowed types: {list(PROMPTS.keys())}"}

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "meta-llama/llama-4-maverick:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": PROMPTS[ocr_type]
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
            })
        )

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                return {"success": True, "result": content}
            else:
                return {"error": "No response from AI model"}
        else:
            return {"error": f"API request failed with status {response.status_code}: {response.text}"}

    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "available_types": list(PROMPTS.keys())})


@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    """Main OCR endpoint"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Check OCR type parameter
        ocr_type = request.form.get('type', '').upper()
        if not ocr_type:
            return jsonify({"error": "OCR type parameter 'type' is required"}), 400

        if ocr_type not in PROMPTS:
            return jsonify({
                "error": f"Invalid OCR type '{ocr_type}'. Allowed types: {list(PROMPTS.keys())}"
            }), 400

        # Validate file
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Allowed extensions: {list(ALLOWED_EXTENSIONS)}"
            }), 400

        # Read and validate file size
        file_bytes = file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            return jsonify({"error": f"File size exceeds maximum limit of {MAX_FILE_SIZE / 1024 / 1024}MB"}), 400

        # Validate that it's actually an image
        try:
            image = Image.open(BytesIO(file_bytes))
            image.verify()  # Verify it's a valid image
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400

        # Encode image to base64
        base64_image = encode_image_from_bytes(file_bytes)

        # Process OCR
        result = process_ocr(base64_image, ocr_type)

        if "error" in result:
            return jsonify(result), 500

        return jsonify({
            "success": True,
            "ocr_type": ocr_type,
            "filename": secure_filename(file.filename),
            "result": result["result"]
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/ocr/base64', methods=['POST'])
def ocr_base64_endpoint():
    """OCR endpoint for base64 encoded images"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON data required"}), 400

        # Check required parameters
        if 'image' not in data or 'type' not in data:
            return jsonify({"error": "Both 'image' (base64) and 'type' parameters are required"}), 400

        base64_image = data['image']
        ocr_type = data['type'].upper()

        # Validate OCR type
        if ocr_type not in PROMPTS:
            return jsonify({
                "error": f"Invalid OCR type '{ocr_type}'. Allowed types: {list(PROMPTS.keys())}"
            }), 400

        # Validate base64 image
        try:
            # Remove data URL prefix if present
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]

            # Decode to validate
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_bytes))
            image.verify()
        except Exception:
            return jsonify({"error": "Invalid base64 image data"}), 400

        # Process OCR
        result = process_ocr(base64_image, ocr_type)

        if "error" in result:
            return jsonify(result), 500

        return jsonify({
            "success": True,
            "ocr_type": ocr_type,
            "result": result["result"]
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == '__main__':
    # Run the app
    print("Starting OCR API Server...")
    print("Available endpoints:")
    print("- GET  /health - Health check")
    print("- POST /ocr - Upload image file")
    print("- POST /ocr/base64 - Send base64 image")
    print("\nAvailable OCR types:", list(PROMPTS.keys()))

    app.run(debug=True, host='0.0.0.0', port=5000)
