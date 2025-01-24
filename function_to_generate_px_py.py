import cv2
import matplotlib.pyplot as plt
import numpy as np
from bbox_generation_function import bb_process


def generate_flatten_indices(raw_image, outer_percentage, inner_percentage):
    
    cropped_image = bb_process(raw_image)   
    
    resize_dim = (1024, 1024) 
    image = cv2.resize(cropped_image, resize_dim)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    h, w = image.shape[:2]  # diameter of the ring
    print(image.shape)
    d = h
    radius = d / 2
    percentage_diff = outer_percentage - inner_percentage
    o_d = int(d * outer_percentage / 100)
    i_d = int(d * inner_percentage / 100)
    outer_r = int(o_d / 2)
    inner_r = int(i_d / 2)
    w = outer_r - inner_r
    num_points = 960  # Target number of points for each row (64 x 960)
    num_radii = 64  # Target number of rows
    image = np.array(image)
    flatten = np.zeros((num_radii, num_points), dtype=np.uint8)
    # Lists to store px and py values of size 64x960
    px_list = np.zeros((num_radii, num_points), dtype=np.int32)
    py_list = np.zeros((num_radii, num_points), dtype=np.int32)
    for idx, r in enumerate(np.linspace(outer_r, inner_r, num_radii)):
        points = [[radius + int(r * np.sin(2 * np.pi * theta / num_points)), 
                   radius + int(r * np.cos(2 * np.pi * theta / num_points))] 
                   for theta in range(num_points)]
        points = np.array(points)
        points = np.clip(points, 0, d - 1)
    
        # Extract x and y coordinates
        px = points[:, 0]  # x coordinates
        py = points[:, 1]  # y coordinates
        
        # Store the coordinates in px_list and py_list
        px_list[idx, :] = np.round(px).astype(int)
        py_list[idx, :] = np.round(py).astype(int)
        
        # Extract pixel values from the image using the coordinates
        t = image[py_list[idx, :], px_list[idx, :]]
        flatten[idx, :] = t
    
    # Save px and py as .npy files
    np.save('testpx.npy', px_list)
    np.save('testpy.npy', py_list)
    
    # Resize the flatten array to the desired dimensions (960 x 64)
    target_size = (960, 64)
    flatten_resized = cv2.resize(flatten, target_size, interpolation=cv2.INTER_LINEAR)
    # mirrored_image = cv2.flip(flatten_resized, 1)
    
    return flatten_resized

