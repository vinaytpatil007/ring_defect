import torch
from torchvision import transforms
from model import create_model
from dataset import DefectSegmentationDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
import glob


px = np.load('/workspace/ring_defect/px.npy')
py = np.load('/workspace/ring_defect/py.npy')


def load_model(model_path):
    # Create model and load the trained weights
    model = create_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def preprocess_image(image_path, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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

    resize_dim = (1024, 1024)  # Ensure consistent dimensions
    image = cv2.resize(cropped_image, resize_dim)


    flatten = image[px, py]

    image = Image.fromarray(flatten)
    image = transform(image)
    image = image.unsqueeze(0)
    return image, flatten
    

def postprocess_output(output):
    # Convert the output to a binary mask (for segmentation tasks)
    output = output.squeeze(0).cpu().detach().numpy()  # Remove batch dimension
    output = (output > 0.5).astype(np.uint8)  # Apply threshold to get binary mask
    return output

def run_inference(model, image_path, transform):
    # Preprocess the image
    image, flatten = preprocess_image(image_path, transform)

    # Run the model
    with torch.no_grad():
        output = model(image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    # Post-process the output
    output = postprocess_output(output)
    return output, flatten

def main():
    # Load the model
    model_path = "/workspace/ring_defect/patch_stack_segmentation/patch_stack_seg_18sept_150epoch.pth"
    model = load_model(model_path)

    # Define the image transforms (same as used in training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Run inference on test images
    test_image_dir = '/workspace/ring_defect/new_dataset/val/*.png'
    image_paths = glob.glob(test_image_dir)

    for image_path in image_paths:
        print(f'Running inference on {image_path}')
        output, flatten = run_inference(model, image_path, transform)
        output = output.squeeze(0)

        # Save or display the result
        result_path = image_path.replace(".png", "_output.png")
        cv2.imwrite(result_path, output * 255)  # Save binary mask as an image
        print(f'Saved result to {result_path}')

if __name__ == '__main__':
    main()
