
from flask import Flask, render_template, request
from PIL import Image
import os
import boto3
import io
import torch
from autocdr import predict, get_pred_array, draw_cdr_countours, get_cdr_using_vertical_height

app = Flask(__name__)

# Define constant paths
STATIC_FOLDER: str = 'static'
RESULT_FOLDER: str = os.path.join(STATIC_FOLDER, 'images', 'results')
MODEL_FOLDER: str = os.path.join(STATIC_FOLDER, 'models/new/')
DEFAULT_FOLDER: str = os.path.join(STATIC_FOLDER, 'images', 'default')

# Configure app settings
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DEFAULT_FOLDER'] = DEFAULT_FOLDER

# # AWS configuration
# AWS_ACCESS_KEY: str = 'XXXXXXXXX'
# AWS_SECRET_KEY: str = 'XXXXXXXXXX'
# BUCKET_NAME: str = 'eazyvitals'


# Initialize S3 client
# s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cup_model_loaded = torch.load(os.path.join(app.config['MODEL_FOLDER'], 'cup_best_model.pth'), map_location=device)
disc_model_loaded = torch.load(os.path.join(app.config['MODEL_FOLDER'], 'disc_best_model.pth'), map_location=device)

def calculate_cdr(image_path: str) -> tuple:
    """
    Calculate Cup-to-Disc Ratio (CDR) for a given image.
    Args:
        image_path (str): Path to the input image.
    Returns:
        Tuple containing calculated CDR, paths to intermediate images, and CDR contours plot.
    """
    cup_pred = predict(cup_model_loaded, image_path)
    disc_pred = predict(disc_model_loaded, image_path)
    
    cup_image_path = os.path.join(app.config['RESULT_FOLDER'], 'pred_cup_image.png')
    disc_image_path = os.path.join(app.config['RESULT_FOLDER'], 'pred_disc_image.png')


    cup_pred_array = get_pred_array(cup_pred, cup_image_path)
    disc_pred_array = get_pred_array(disc_pred, disc_image_path)

    overlay_image_path = os.path.join(app.config['RESULT_FOLDER'], 'overlay_image.png')
    cdr_contours_image_path = os.path.join(app.config['RESULT_FOLDER'], 'cdr_contours_image_path.png')
    # plot_overlay_image(image_path, cup_pred_array, disc_pred_array, save_path=overlay_image_path)
    contour_fig = draw_cdr_countours(image_path, cup_pred_array, disc_pred_array, save_path=cdr_contours_image_path)
    
    # pred_cdr = get_cdr(cup_pred_array, disc_pred_array)
    pred_cdr_vertical = get_cdr_using_vertical_height(cup_pred_array, disc_pred_array)
    
    return round(pred_cdr_vertical, 2), cdr_contours_image_path

@app.route('/')
def index() -> str:
    """Render the main index page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload() -> str:
    """Handle image upload and CDR calculation."""
    try:
        file = request.files['image']
        image_data = file.read()
        
        original_image = Image.open(io.BytesIO(image_data))
        resized_image = original_image.resize((512, 512))
        
        original_image_path = os.path.join(app.config['MODEL_FOLDER'], 'original_image.png')
        resized_image_path = os.path.join(app.config['MODEL_FOLDER'], 'resized_image.png')
        
        original_image.save(original_image_path)
        resized_image.save(resized_image_path)
        
        cdr_vertical, cdr_contours_image_path = calculate_cdr(original_image_path)
        
        return render_template(
            'result.html',
            cdr_vertical=cdr_vertical,
            cdr_contours_image_path=cdr_contours_image_path,
        )
    except Exception as e:
        error_message = "Sorry, could not process image. Please upload a retinal image."
        return render_template(
            'result.html',
            cdr_vertical=error_message
        )

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
