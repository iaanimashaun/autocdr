
# import random
# import os
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision import transforms as T
# import numpy as np
# import cv2

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def predict(model, image_path):
#     model.eval()
    
#     img = Image.open(image_path)
#     transform = T.ToTensor()
#     ig = transform(img)
#     with torch.no_grad():
#         pred = model([ig.to(device)])
#     return pred

# def get_pred_array(pred, save_path=""):
#     pred_array = (pred[0]["masks"][0].cpu().detach().numpy() * 255).astype("uint8").squeeze()
#     if save_path:
#         plt.imsave(save_path, pred_array)
#     return pred_array

# def get_cdr(cup_array,disc_array):
        
#     cup_pixels = np.count_nonzero(cup_array)
#     disc_pixels = np.count_nonzero(disc_array)
#     cdr = cup_pixels / disc_pixels
#     return cdr

# def calculate_vertical_height(mask_array):
#     # Find the indices of non-zero elements (coordinates of the pixels)
#     non_zero_indices = np.transpose(np.nonzero(mask_array))
    
#     # Find the minimum and maximum y-coordinates (vertical positions)
#     min_y = np.min(non_zero_indices[:, 0])
#     max_y = np.max(non_zero_indices[:, 0])
    
#     # Calculate the vertical height as the difference between max and min y-coordinates
#     vertical_height = max_y - min_y + 1  # Adding 1 to include both endpoints
#     return vertical_height

# def get_cdr_using_vertical_height(cup_array, disc_array):
#     cup_vertical_height = calculate_vertical_height(cup_array)
#     disc_vertical_height = calculate_vertical_height(disc_array)
    
#     cdr = cup_vertical_height / disc_vertical_height
#     return cdr


# def plot_cup_disc(image_path, cup_pred_array, disc_pred_array):
          
#     img = Image.open(image_path)
#     original_image_array = np.array(img)
  
#     ax1 = plt.subplot(2,2,1)
#     ax2 = plt.subplot(2,2,2)
#     # ax3 = plt.subplot(2,2,3)
#     # ax4 = plt.subplot(2,2,4)

#     ax1.imshow(cup_pred_array)
#     ax1.set_title("predicted cup")
#     ax2.imshow(disc_pred_array)
#     ax2.set_title("predicted disc")
#     # ax3.imshow(np.array(Image.open(cup_mask_path)))
#     # ax3.set_title("ground truth cup")
#     # ax4.imshow(np.array(Image.open(disc_mask_path)))
#     # ax4.set_title("ground truth disc")
    
#     plt.tight_layout()
#     plt.show()



# def plot_overlay_image(image_path, cup_pred_array, disc_pred_array, save_path=""):
           
            
#     img = Image.open(image_path)
#     original_image_array = np.array(img)
#     # Create a copy of the original image
#     overlayed_image_copy = np.copy(original_image_array)

#     # Define overlay color (e.g., red color)
#     overlay_color_cup = [255, 0, 0]  # Red color [R, G, B]
#     overlay_color_disc = [255, 255, 255]  # White color [R, G, B]


#     # Apply the predicted mask as overlay
#     overlayed_image_copy[disc_pred_array != 0] = overlay_color_disc
#     overlayed_image_copy[cup_pred_array != 0] = overlay_color_cup

#     if save_path:
#         plt.imsave(save_path, overlayed_image_copy)

#     # Display the overlayed image using Matplotlib
#     plt.figure(figsize=(8, 8))
#     plt.imshow(overlayed_image_copy)
#     plt.axis('off')
#     plt.show()    
           
           
# # def draw_cdr_countours_old(image_path, cup_pred_array, disc_pred_array, save_path="" ):

# #     img = Image.open(image_path)
# #     original_image_array = np.array(img)

# #     # Find contours in the masks
# #     disc_contours, _ = cv2.findContours(disc_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     cup_contours, _ = cv2.findContours(cup_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     # Draw contours on a copy of the original image
# #     original_image_copy = original_image_array.copy()
# #     cv2.drawContours(original_image_copy, disc_contours, -1, (0, 0, 255), 2)  # Red outline for disc
# #     cv2.drawContours(original_image_copy, cup_contours, -1, (255, 255, 255), 2)  # White outline for cup

# #     # Display the overlayed image using Matplotlib
# #     plt.imsave(save_path, original_image_copy)
# #     plt.imshow(original_image_copy)
# #     plt.axis('off')
# #     plt.show()
    
    
    
    
    
# # def draw_cdr_countours(image_path, cup_pred_array, disc_pred_array, save_path=""):
# #     img = Image.open(image_path)
# #     original_image_array = np.array(img)

# #     fig, ax = plt.subplots(figsize=(8, 6))

# #     # Find contours in the masks
# #     disc_contours, _ = cv2.findContours(disc_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     cup_contours, _ = cv2.findContours(cup_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     # Draw contours on a copy of the original image
# #     original_image_copy = original_image_array.copy()

# #     arrow_length = 10.0  # Adjust the arrow length as needed
# #     rotation_angle = 0  # Adjust the rotation angle in degrees

# #     if len(disc_contours) > 0:
# #         disc_contour = disc_contours[0]  # Assuming there's only one disc contour
# #         cv2.drawContours(original_image_copy, [disc_contour], -1, (0, 0, 255), 2)  # Red outline for disc
# #         start_point = tuple(disc_contour[0][0])
# #         end_point = tuple(disc_contour[-1][0])
# #         arrow_direction = np.array(start_point) - np.array(end_point)  # Reverse the direction for disc
# #         # arrow_direction /= np.linalg.norm(arrow_direction)
        
# #         # Rotate the arrow direction vector by the specified angle
# #         rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
# #                                     [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])
# #         rotated_arrow_direction = np.dot(rotation_matrix, arrow_direction)

# #         arrow_tip = np.array(start_point) + rotated_arrow_direction * arrow_length

# #         ax.annotate("Optic Disc", end_point, color='blue', fontsize=15,
# #                      xytext=tuple(arrow_tip),
# #                      arrowprops=dict(facecolor='blue', arrowstyle="->"))

# #     rotation_angle = 45  # Adjust the rotation angle in degrees
# #     arrow_length = 3.0  # Adjust the arrow length as needed

# #     if len(cup_contours) > 0:
# #         cup_contour = cup_contours[0]  # Assuming there's only one cup contour
# #         cv2.drawContours(original_image_copy, [cup_contour], -1, (255, 0, 0), 2)  # Red outline for cup
# #         start_point = tuple(cup_contour[0][0])
# #         end_point = tuple(cup_contour[-1][0])
# #         arrow_direction = np.array(end_point) - np.array(start_point)  # Keep the direction for cup
# #         # arrow_direction /= np.linalg.norm(arrow_direction)
        
# #         # Rotate the arrow direction vector by the specified angle
# #         rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
# #                                     [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])
# #         rotated_arrow_direction = np.dot(rotation_matrix, arrow_direction)

# #         arrow_tip = np.array(end_point) + rotated_arrow_direction * arrow_length

# #         ax.annotate("Optic Cup", end_point, color='red', fontsize=15,
# #                      xytext=tuple(arrow_tip),
# #                      arrowprops=dict(facecolor='red', arrowstyle="->"))

# #     # Display the overlayed image
# #     ax.imshow(original_image_copy)
# #     ax.axis('off')

# #     # Save the figure
# #     if save_path:
# #         fig.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    
# #     # Show the image
# #     plt.show()






# import numpy as np
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt

# def draw_cdr_countours(image_path, cup_pred_array, disc_pred_array, save_path=""):
#     img = Image.open(image_path)
#     original_image_array = np.array(img)

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Find contours in the masks
#     disc_contours, _ = cv2.findContours(disc_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cup_contours, _ = cv2.findContours(cup_pred_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours on a copy of the original image
#     original_image_copy = original_image_array.copy()

#     arrow_length = 3.0  # Adjust the arrow length as needed
#     rotation_angle = 60  # Adjust the rotation angle in degrees

#     if len(disc_contours) > 0:
#         disc_contour = disc_contours[0]  # Assuming there's only one disc contour
#         cv2.drawContours(original_image_copy, [disc_contour], -1, (0, 0, 255), 2)  # Red outline for disc
#         start_point = tuple(disc_contour[0][0])
#         end_point = tuple(disc_contour[-1][0])
#         arrow_direction = np.array(start_point) - np.array(end_point)  # Reverse the direction for disc
#         # arrow_direction /= np.linalg.norm(arrow_direction)
        
#         # Rotate the arrow direction vector by the specified angle
#         rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
#                                     [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])
#         rotated_arrow_direction = np.dot(rotation_matrix, arrow_direction)

#         arrow_tip = np.array(start_point) + rotated_arrow_direction * arrow_length

#         ax.annotate("Optic Disc", start_point, color='blue', fontsize=10,
#                      xytext=tuple(arrow_tip),
#                      arrowprops=dict(facecolor='blue', arrowstyle="->"))

#     if len(cup_contours) > 0:
#         cup_contour = cup_contours[0]  # Assuming there's only one cup contour
#         cv2.drawContours(original_image_copy, [cup_contour], -1, (255, 0, 0), 2)  # Red outline for cup
#         start_point = tuple(cup_contour[0][0])
#         end_point = tuple(cup_contour[-1][0])
#         arrow_direction = np.array(end_point) - np.array(start_point)  # Keep the direction for cup
#         # arrow_direction /= np.linalg.norm(arrow_direction)
        
#         # Rotate the arrow direction vector by the specified angle
#         rotation_matrix = np.array([[np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
#                                     [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))]])
#         rotated_arrow_direction = np.dot(rotation_matrix, arrow_direction)

#         arrow_tip = np.array(end_point) + rotated_arrow_direction * arrow_length

#         ax.annotate("Optic Cup", end_point, color='red', fontsize=10,
#                      xytext=tuple(arrow_tip),
#                      arrowprops=dict(facecolor='red', arrowstyle="->"))

#     # Display the overlayed image
#     ax.imshow(original_image_copy)
#     ax.axis('off')

#     # Save the figure
#     # if save_path:
#     # fig.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    
#     # Show the image
#     plt.show()
    
#     return fig

    

# # Example usage
# # draw_cdr_countours("image.jpg", cup_pred_array, disc_pred_array, "output_image.jpg")



























import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms as T


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








