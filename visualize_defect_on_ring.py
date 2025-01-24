import cv2
import torch
import numpy as np
from PIL import Image
from model import create_model
from torchvision import transforms

px = np.load('C:/ring_defect/px.npy')
py = np.load('C:/ring_defect/py.npy')

def load_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def bb_process(image, min_contour_area=1000):   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)    
    contours, _ = cv2.findContours(morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        resize_dim = (1024, 1024) 
        cropped_image = cv2.resize(cropped_image, resize_dim)
        return cropped_image

def preprocess_image(image, transform):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = bb_process(image)
    bb_image = image.copy()
    flatten = image[px, py]
    image = Image.fromarray(flatten)    
    image = transform(image)
    image = image.unsqueeze(0)  
    return image, flatten, bb_image

def postprocess_output(output):
    output = output.squeeze(0).cpu().detach().numpy()  
    output = (output > 0.5).astype(np.uint8) 
    return output

def run_inference(model, image, transform):
    image, flatten, bb_image = preprocess_image(image, transform)
    with torch.no_grad():
        output = model(image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    output = postprocess_output(output)
    return output, flatten, bb_image

def visualize_ring(image, model_path):
    model = load_model(model_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    output, flatten, bb_image = run_inference(model, image, transform)
    new_flatten = flatten.copy()
    output = output.squeeze(0)
    bb_image1 = bb_image.copy()
    radius = bb_image.shape[0]//2
    bb_image = cv2.cvtColor(bb_image, cv2.COLOR_BGR2RGB)
    center = (bb_image.shape[1] // 2, bb_image.shape[0] // 2)
    column_sums = np.sum(output, axis=0)
    regions_cords_c = []
    cords_c = []
    for i, c in enumerate(column_sums):
        if c>=1 :
            cords_c.append(i)
        else:
            if len(cords_c) > 2:
                regions_cords_c.append(cords_c)
            cords_c = []

    if len(cords_c) > 2:
        regions_cords_c.append(cords_c)

    for k in range(len(regions_cords_c)):
        cords_c = regions_cords_c[k]
        flatten_patch = output[:, min(cords_c):max(cords_c)]
        regions_cords_r = []
        cords_r = []
        for r in range(len(flatten_patch)):
            if 1 in flatten_patch[r]:
                cords_r.append(r)
        y1 = min(cords_r)
        y2 = max(cords_r)
        x1 = min(cords_c)
        x2 = max(cords_c)
        defect_positions = [(x1, y1), (x2, y2)]
        original_positions = [(int(py[y, x]), int(px[y, x])) for x, y in defect_positions]
        
        x1_bb = original_positions[0][0]
        y1_bb = original_positions[0][1]
        x2_bb = original_positions[1][0]
        y2_bb = original_positions[1][1]
                
        x1_bb, x2_bb = min(x1_bb, x2_bb), max(x1_bb, x2_bb)
        y1_bb, y2_bb = min(y1_bb, y2_bb), max(y1_bb, y2_bb)
        
        # print(x1_bb, x2_bb, y1_bb, y2_bb)
        cv2.rectangle(bb_image1, (x1_bb-20, y1_bb-20), (x2_bb+20, y2_bb+20), (255, 0, 0), 4)
                
    return bb_image, bb_image1   