import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import uuid # To generate unique filenames

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
# Directory to store uploaded images
UPLOAD_FOLDER = 'uploads'
# Directory to store predicted images (with annotations)
PREDICTED_IMAGES_FOLDER = 'static/predicted_images'
# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload and predicted images folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_IMAGES_FOLDER, exist_ok=True)

# --- YOLO Model Loading ---
# Adjust this path if your best.pt is not in the same directory as app.py
MODEL_PATH = 'models\\best.pt' 

# Verify the model path exists before loading
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights not found at {MODEL_PATH}. Please ensure 'best.pt' is in the same directory as app.py or update the MODEL_PATH variable.")
    model = None # Set model to None to handle gracefully
else:
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH) # Load your trained model

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main upload form."""
    return render_template('index.html', predicted_image_name=None, detections=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, runs prediction, and displays results."""
    if model is None:
        return render_template('index.html', error="ML model could not be loaded. Please check server logs.")

    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        # Generate a unique filename to prevent overwrites
        original_filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + '_' + original_filename
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        print(f"Starting prediction on: {filepath}")
        try:
            # Run prediction. YOLO will create a new 'exp' directory for each run.
            # We don't use the 'name' parameter to avoid issues with existing directories.
            results = model.predict(
                source=filepath,
                save=True,          # Save the predicted image with bounding boxes and labels
                conf=0.5,           # Confidence threshold
                iou=0.7,            # IoU threshold for NMS
                # verbose=False # Set to True for more detailed console output from YOLO
            )

       
            if results and len(results) > 0:
                save_dir = results[0].save_dir # e.g., 'runs/detect/exp' or 'runs/detect/exp2'
                predicted_image_path = os.path.join(save_dir, unique_filename)
                
                predicted_image_name_for_url = 'predicted_' + unique_filename
                final_predicted_image_path = os.path.join(PREDICTED_IMAGES_FOLDER, predicted_image_name_for_url)
                
                # Ensure the predicted image actually exists before moving
                if os.path.exists(predicted_image_path):
                    os.rename(predicted_image_path, final_predicted_image_path)
                else:
                    return render_template('index.html', error=f"Predicted image not found at {predicted_image_path}")

                # Process detection results for display
                detections = []
                for r in results:
                    if r.boxes: # Check if there are any detections in this frame/image
                        for box in r.boxes:
                            # x1, y1, x2, y2 = box.xyxy[0].tolist() # Bounding box coordinates
                            confidence = box.conf[0].tolist()    # Confidence score
                            class_id = box.cls[0].tolist()       # Class ID
                            class_name = model.names[int(class_id)] # Get human-readable class name
                            detections.append({
                                'class_name': class_name,
                                'confidence': f"{confidence:.2f}"
                                # 'box': [int(x1), int(y1), int(x2), int(y2)] # Can add this if needed
                            })
                    else:
                        detections.append({'message': 'No detections in this image.'})
                
                # Clean up the original uploaded file
                os.remove(filepath)

                return render_template('index.html', 
                                       predicted_image_name=predicted_image_name_for_url, 
                                       detections=detections)
            else:
                # Clean up the original uploaded file
                os.remove(filepath)
                return render_template('index.html', error='No prediction results returned.')

        except Exception as e:
            print(f"Prediction error: {e}")
            # Clean up the original uploaded file in case of error
            os.remove(filepath)
            return render_template('index.html', error=f'An error occurred during prediction: {e}')
    
    return render_template('index.html', error='Invalid file type.')

@app.route('/static/predicted_images/<filename>')
def display_predicted_image(filename):
    """Serves the predicted images from the static folder."""
    return send_from_directory(PREDICTED_IMAGES_FOLDER, filename)

if __name__ == '__main__':
    # Run the Flask app
    # In a production environment, use a WSGI server like Gunicorn or uWSGI
    app.run(debug=True) # debug=True enables reloader and debugger
