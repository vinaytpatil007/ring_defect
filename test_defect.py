import random
import os
import glob
import cv2
import matplotlib.pyplot as plt
from visualize_defect_on_ring import *


model_path = "/home/vinay/ring_defect/patch_stack_segmentation/Models/patch_stack_seg_18sept_150epoch.pth" 
test_image_dir = '/home/vinay/ring_defect/patch_stack_segmentation/dataset/rough_imgs/ok/*.png'
image_paths = glob.glob(test_image_dir)
for count, image_path in enumerate(image_paths):
    
#     image_path = random.choice(image_paths)
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    print(image_name)
    bb_image, bb_image1 = visualize_ring(image, model_path)

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(bb_image)
    plt.title('BB_image')
    plt.axis('off')

#     plt.show()
    plt.subplot(122)
    plt.imshow(bb_image1)
    plt.title('Predicted_BB_image')
    plt.axis('off')
    # plt.savefig(f'/home/vinay/ring_defect/patch_stack_segmentation/ring_prediction_images/test_images_prediction/{image_name}.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f'{count}_: {image_name} image is saved')