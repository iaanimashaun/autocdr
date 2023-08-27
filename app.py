# from flask import Flask, render_template, request, jsonify
# import boto3
# import io
# from PIL import Image
# import torch
# import os
# import matplotlib.pyplot as plt
# from cdr import predict, get_pred_array, get_cdr, plot_overlay_image, draw_cdr_countours, get_cdr_using_vertical_height
# app = Flask(__name__)


# RESULT_FOLDER = os.path.join('static', 'images/results')
# MODEL_FOLDER = os.path.join('static', 'models')
# IMAGES_FOLDER = os.path.join('static', 'images')
# DEFAULT_FOLDER = os.path.join('static', 'images/default')

# app.config['RESULT_FOLDER'] = RESULT_FOLDER
# app.config['MODEL_FOLDER'] = MODEL_FOLDER
# app.config['IMAGES_FOLDER'] = IMAGES_FOLDER
# app.config['DEFAULT_FOLDER'] = DEFAULT_FOLDER

# # AWS configuration
# AWS_ACCESS_KEY = 'AKIA5TT3AHP4H7FEPBHF'
# AWS_SECRET_KEY = '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'
# BUCKET_NAME = 'eazyvitals'

# # Initialize S3 client
# s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)



# default_image_urls = [
#         # url_for('static', filename='default_images/image1.jpg'),
#         # url_for('static', filename='default_images/image2.jpg'),
#         os.path.join(app.config['DEFAULT_FOLDER'], 'CRFO-v4-1.png'),
#         os.path.join(app.config['DEFAULT_FOLDER'], 'CRFO-v4-3.png'),
#         os.path.join(app.config['DEFAULT_FOLDER'], 'CRFO-v4-4.png')
# # /home/ubuntu/projects/glaucoma/backend/static/images/default/CRFO-v4-1.png
#         # Add more image URLs as needed
#     ]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # cup_model_path = '/home/ubuntu/projects/glaucoma/backend/static/models/cup_model.pth'  # Replace this with the path where the model is saved
# # disc_model_path = os.path.join(app.static_folder, "models", "disc_model.pth") # Replace this with the path where the model is saved

# cup_model_path = os.path.join(app.config['MODEL_FOLDER'], 'cup_best_model.pth')
# disc_model_path = os.path.join(app.config['MODEL_FOLDER'], 'disc_best_model.pth')

# # Load the entire model from the saved file
# cup_model_loaded = torch.load(cup_model_path,  map_location=device)
# disc_model_loaded = torch.load(disc_model_path,  map_location=device)




# def calculate_cdr(image_path):
       
#         cup_pred = predict(cup_model_loaded, image_path)
#         disc_pred = predict(disc_model_loaded, image_path)

#         # print("cup pred type: ", type(cup_pred))
#         # print("cup pred : ", (cup_pred))
#         # print("disc_pred pred : ", (disc_pred))
        
        
#         cup_image_path = os.path.join(app.config['RESULT_FOLDER'], 'pred_cup_image.png')
#         disc_image_path = os.path.join(app.config['RESULT_FOLDER'], 'pred_disc_image.png')

#         cup_pred_array = get_pred_array(cup_pred, cup_image_path)
#         disc_pred_array = get_pred_array(disc_pred, disc_image_path)


#         overlay_image_path = os.path.join(app.config['RESULT_FOLDER'], 'overlay_image.png')
#         cdr_contours_image_path = os.path.join(app.config['RESULT_FOLDER'], 'cdr_contours_image_path.png')
#         _ = plot_overlay_image(image_path, cup_pred_array, disc_pred_array, save_path=overlay_image_path)
#         contour_fig = draw_cdr_countours(image_path, cup_pred_array, disc_pred_array, save_path=cdr_contours_image_path)
        
#         contour_fig.savefig(cdr_contours_image_path, bbox_inches='tight', pad_inches=0, transparent=True)

#         # plt.axis('off')
#         # plt.imsave(save_path, cup_pred_array)
#         # plt.savefig(os.path.join(app.static_folder, "images", "pred_disc_image.jpg"), bbox_inches='tight')
#         # plt.close()
        
#         pred_cdr = get_cdr(cup_pred_array,disc_pred_array)
#         pred_cdr_vertical = get_cdr_using_vertical_height(cup_pred_array,disc_pred_array)
#         print(f"CDR PRED = {pred_cdr} ")
        
#         return round(pred_cdr, 2), round(pred_cdr_vertical, 2), cup_image_path, disc_image_path, overlay_image_path, cdr_contours_image_path
   

# @app.route('/')
# def index():
#     # return render_template('index.html')
#     return render_template(
#             'index.html',
#             # default_image_urls_1=default_image_urls[0],
#             # default_image_urls_2=default_image_urls[1],
#             # default_image_urls_3=default_image_urls[2]
#         )




# @app.route('/upload', methods=['POST'])
# def upload():
#     # title = request.form['title']
#     file = request.files['image']
#     print("file: ", file)
#     print("dir: ", dir(file))
#     try:
#         # Read the image data
#         image_data = file.read()
#         # Resize the image to 512x512
#         original_image = Image.open(io.BytesIO(image_data))
#         resized_image = original_image.resize((512, 512))
#         original_image_path = os.path.join(app.config['MODEL_FOLDER'], 'original_image.png')
#         resized_image_path = os.path.join(app.config['MODEL_FOLDER'], 'resized_image.png')

#         # Save the resized image to a file
#         original_image.save(original_image_path)
#         resized_image.save(resized_image_path)

#         # Upload the resized image to S3
#         # s3.upload_file(original_image_path, BUCKET_NAME, file.filename)
#         # s3.upload_file(resized_image_path, BUCKET_NAME, file.filename)

#         # Calculate cup-disc ratio for the resized image
#         cdr, cdr_vertical, cup_image_path, disc_image_path, overlay_image_path, cdr_contours_image_path = calculate_cdr(original_image_path)





#     # # try:
#     #     image_data = file.read()
#     #     s3.upload_fileobj(io.BytesIO(image_data), BUCKET_NAME, file.filename)
        
#     #     # Calculate cup-disc ratio
#     #     cdr, cup_image_path, disc_image_path, overlay_image_path, cdr_contours_image_path = calculate_cdr(io.BytesIO(image_data))
        
#         print("cdr: ", cdr)
#         print("cup_image_path: ", cup_image_path)
#         print("disc_image_path: ", disc_image_path)
#         print("overlay_image_path: ", overlay_image_path)
        
#         # return render_template('result.html', cdr=cdr, cup_image_path=cup_image_path, disc_image_path=disc_image_path, overlay_image_path=overlay_image_path, cdr_contours_image_path=cdr_contours_image_path)
#         return render_template(
#             'result.html',
#             cdr=cdr,
#             cdr_contours_image_path=cdr_contours_image_path,
#             # filename=file.filename,
#             cdr_vertical=cdr_vertical,
#             # default_image_urls=default_image_urls
#         )
#     except Exception as e:
#             # print("An error occurred:", e)
            
#             print("Sorry, could not process retinal image:")
#             print("Likely cause: It seems the uploaded image is not a retina image.")
#             error_message = "Sorry, could not process image. Please upload a retinal image"
#             # return "Likely cause: It seems the uploaded image is not a retina image."
#             return render_template(
#             'result.html',
#             # cdr=cdr,
#             # cdr_contours_image_path=cdr_contours_image_path,
#             # filename=file.filename,
#             cdr_vertical=error_message,
#             # default_image_urls=default_image_urls
#         )

#     # except Exception as e:
#     #     return jsonify({"error": "Error uploading image: " + str(e)}), 500


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", debug=True)


































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
MODEL_FOLDER: str = os.path.join(STATIC_FOLDER, 'models')
DEFAULT_FOLDER: str = os.path.join(STATIC_FOLDER, 'images', 'default')

# Configure app settings
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['DEFAULT_FOLDER'] = DEFAULT_FOLDER

# # AWS configuration
# AWS_ACCESS_KEY: str = 'AKIA5TT3AHP4H7FEPBHF'
# AWS_SECRET_KEY: str = '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'
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
