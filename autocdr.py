
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image, ImageFilter, ImageOps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model: torch.nn.Module, image_path: str) -> torch.Tensor:
    """
    Make predictions using the given model on the input image.
    
    Args:
        model (torch.nn.Module): Trained model for prediction.
        image_path (str): Path to the input image.
        
    Returns:
        torch.Tensor: Predicted tensor.
    """
    model.eval()
    img = Image.open(image_path)
    # img = preprocess_image(image_path)
    transform = T.ToTensor()
    img_tensor = transform(img).to(device)
    with torch.no_grad():
        pred = model([img_tensor])
    return pred


def get_pred_array(pred: dict, save_path: str = "") -> np.ndarray:
    """
    Convert the predicted masks to an image array and save if specified.
    
    Args:
        pred (dict): Predicted mask dictionary.
        save_path (str, optional): Path to save the image array.
        
    Returns:
        np.ndarray: Predicted mask as an image array.
    """
    pred_array = (pred[0]["masks"][0].cpu().detach().numpy() * 255).astype("uint8").squeeze()
    if save_path:
        plt.imsave(save_path, pred_array)
    return pred_array

def get_cdr(cup_array: np.ndarray, disc_array: np.ndarray) -> float:
    """
    Calculate the Cup-to-Disc Ratio (CDR).
    
    Args:
        cup_array (np.ndarray): Binary mask of the optic cup.
        disc_array (np.ndarray): Binary mask of the optic disc.
        
    Returns:
        float: Calculated CDR value.
    """
    cup_pixels = np.count_nonzero(cup_array)
    disc_pixels = np.count_nonzero(disc_array)
    cdr = cup_pixels / disc_pixels
    return cdr

def calculate_vertical_height(mask_array: np.ndarray) -> int:
    """
    Calculate the vertical height of a binary mask.
    
    Args:
        mask_array (np.ndarray): Binary mask array.
        
    Returns:
        int: Vertical height of the mask.
    """
    non_zero_indices = np.transpose(np.nonzero(mask_array))
    min_y = np.min(non_zero_indices[:, 0])
    max_y = np.max(non_zero_indices[:, 0])
    vertical_height = max_y - min_y + 1
    return vertical_height

def get_cdr_using_vertical_height(cup_array: np.ndarray, disc_array: np.ndarray) -> float:
    """
    Calculate CDR using vertical height of the cup and disc.
    
    Args:
        cup_array (np.ndarray): Binary mask of the optic cup.
        disc_array (np.ndarray): Binary mask of the optic disc.
        
    Returns:
        float: Calculated CDR value using vertical height.
    """
    cup_vertical_height = calculate_vertical_height(cup_array)
    disc_vertical_height = calculate_vertical_height(disc_array)
    cdr = cup_vertical_height / disc_vertical_height
    return cdr

def draw_cdr_countours(image_path: str, cup_pred_array: np.ndarray, disc_pred_array: np.ndarray, save_path: str = "") -> plt.Figure:
    """
    Draw the contours of the optic disc and optic cup on the original image.
    
    Args:
        image_path (str): Path to the original image.
        cup_pred_array (np.ndarray): Predicted mask for optic cup.
        disc_pred_array (np.ndarray): Predicted mask for optic disc.
        save_path (str, optional): Path to save the output image with contours.
        
    Returns:
        plt.Figure: Matplotlib figure containing the image with drawn contours.
    """
    img = Image.open(image_path)
    original_image_array = np.array(img)

    fig, ax = plt.subplots(figsize=(10, 6))

    disc_contours, _ = cv2.findContours(disc_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_contours, _ = cv2.findContours(cup_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    original_image_copy = original_image_array.copy()

    arrow_length = 3.0
    rotation_angle = 60

    if len(disc_contours) > 0:
        disc_contour = disc_contours[0]
        cv2.drawContours(original_image_copy, [disc_contour], -1, (0, 0, 255), 2)
        start_point = tuple(disc_contour[0][0])
        end_point = tuple(disc_contour[-1][0])
        arrow_direction = np.array(start_point) - np.array(end_point)
        rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
                                    [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])
        rotated_arrow_direction = np.dot(rotation_matrix, arrow_direction)
        arrow_tip = np.array(start_point) + rotated_arrow_direction * arrow_length

        ax.annotate("Optic Disc", start_point, color='blue', fontsize=10,
                     xytext=tuple(arrow_tip),
                     arrowprops=dict(facecolor='blue', arrowstyle="->"))

    if len(cup_contours) > 0:
        cup_contour = cup_contours[0]
        cv2.drawContours(original_image_copy, [cup_contour], -1, (255, 0, 0), 2)
        start_point = tuple(cup_contour[0][0])
        end_point = tuple(cup_contour[-1][0])
        arrow_direction = np.array(end_point) - np.array(start_point)
        rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
                                    [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])
        rotated_arrow_direction = np.dot(rotation_matrix, arrow_direction)
        arrow_tip = np.array(end_point) + rotated_arrow_direction * arrow_length

        ax.annotate("Optic Cup", end_point, color='red', fontsize=10,
                     xytext=tuple(arrow_tip),
                     arrowprops=dict(facecolor='red', arrowstyle="->"))

    ax.imshow(original_image_copy)
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)

    plt.show()
    return fig




def plot_overlay_image(image_path, cup_pred_array, disc_pred_array, save_path="") -> plt.Figure:
    """
    Plot the original image overlayed with predicted optic disc and optic cup masks.
    
    Args:
        image_path (str): Path to the original image.
        cup_pred_array (np.ndarray): Predicted mask for optic cup.
        disc_pred_array (np.ndarray): Predicted mask for optic disc.
        save_path (str, optional): Path to save the overlayed image.
        
    Returns:
        plt.Figure: Matplotlib figure containing the overlayed image.
    """     
    img = Image.open(image_path)
    original_image_array = np.array(img)
    # Create a copy of the original image
    overlayed_image_copy = np.copy(original_image_array)

    # Define overlay color (e.g., red color)
    overlay_color_cup = [255, 0, 0]  # Red color [R, G, B]
    overlay_color_disc = [255, 255, 255]  # White color [R, G, B]


    # Apply the predicted mask as overlay
    overlayed_image_copy[disc_pred_array != 0] = overlay_color_disc
    overlayed_image_copy[cup_pred_array != 0] = overlay_color_cup

    if save_path:
        plt.imsave(save_path, overlayed_image_copy)

    # Display the overlayed image using Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(overlayed_image_copy)
    plt.axis('off')
    plt.show()  







def preprocess_image(image_path):
    # Load the raw retinal image using PIL
    raw_image = Image.open(image_path).convert("L")  # Convert to grayscale

    # Apply histogram equalization for contrast enhancement
    equalized_image = ImageOps.equalize(raw_image)

    # Apply Gaussian filtering for noise reduction
    filtered_image = equalized_image.filter(ImageFilter.GaussianBlur(radius=2))

    return filtered_image
