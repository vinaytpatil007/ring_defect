import cv2
import glob
import json
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DefectSegmentationDataset(Dataset):
    def __init__(self, img_path, transform, px, py):
        self.transform = transform
        self.img_path = img_path
        self.px = px
        self.py = py

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        image_paths = glob.glob(self.img_path)
        image_path = random.choice(image_paths)
        json_path = image_path[:-4] + '.geojson'
        image = cv2.imread(image_path)
        mask = np.zeros((image.shape[1], image.shape[0]))
        try:
            with open(json_path) as f:
                data = json.load(f)
            for feature in data['features']:
                coordinates_0 = feature['geometry']['coordinates'][0]
                polygon_0 = np.array(coordinates_0, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [polygon_0], color=[1])
        except Exception as e:
            print(e)
                    
        min_contour_area = 1000
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped_image = image[y : y + h, x : x + w]
            cropped_mask = mask[y : y + h, x : x + w]

        resize_dim = (1024, 1024)  # Ensure consistent dimensions
        image = cv2.resize(cropped_image, resize_dim)
        mask = cv2.resize(cropped_mask, resize_dim)
        mask = (mask>0.5).astype(float)
        
        flatten = image[self.px, self.py]
        mask = mask[self.px, self.py]

        image = Image.fromarray(flatten)

        new_image = np.zeros((256, 256, 3), dtype=np.uint8)
        new_mask = np.zeros((256, 256), dtype=np.uint8)

        flatten_w, flatten_h = image.size
        start_points = [random.randint(0, flatten_w - 256) for _ in range(4)]
        for idx, start_point in enumerate(start_points):
            patch = np.array(image)[:, start_point : start_point + 256]
            new_image[idx * 64 : idx * 64 + 64, :] = patch
            resized_mask = cv2.resize(mask[:, start_point : start_point + 256], (256, 64))
            new_mask[idx * 64 : idx * 64 + 64, :] = resized_mask

        new_mask = (new_mask>0.5).astype(float)
        new_image = Image.fromarray(new_image.astype("uint8"))
        new_image = self.transform(new_image)
        new_mask = torch.tensor(new_mask)
        return new_image, new_mask
