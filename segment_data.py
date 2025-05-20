import pandas as pd
from tqdm import tqdm
import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

# Step 1: Load the dataset labels
train_df = pd.read_csv('final_data/Train/train.csv')
val_df = pd.read_csv('final_data/Test/test.csv')
test_df = pd.read_csv('final_data/Val/val.csv')

IMAGE_LIST = []

# Step 2: load dataset images !
def get_image_path(dir_path):
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    image = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and os.path.splitext(f)[1].lower() in image_extensions][0]
    return dir_path+ '/' +image

train_images = []
for i in tqdm(range(len(train_df)), desc='Loading train images'):
        path = f'final_data/Train/Posts/trn-{i+1}/'
        path = get_image_path(path)
        train_images.append(path)
        
test_images = []
for i in tqdm(range(len(test_df)), desc='Loading test images'):
        path = f'final_data/Test/Posts/tst-{i+1}/'
        path = get_image_path(path)
        test_images.append(path)

HOME = os.getcwd()
print("HOME:", HOME)

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

# Step 3: Set device and model type
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Step 4: Load the SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# Step 5: Load the image
OUTPUT_BASE_DIR = "sam/total_data"
def get_segments(IMAGE_PATH):
    split = 'test' if 'Test' in IMAGE_PATH else 'train'
    
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Step 6: Generate masks for the image
    sam_result = mask_generator.generate(image_rgb)
    id_ = IMAGE_PATH.split('-')[1].split('/')[0]
    output_dir = os.path.join(OUTPUT_BASE_DIR, split, f"{'tst' if split == 'test' else 'trn'}-{id_}")
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(sorted(sam_result, key=lambda x: x['area'], reverse=True), start=1):
        if i == 11:
            break
        segment = mask['segmentation']
        segment_filename = os.path.join(output_dir, f"segment-{i}.png")
        
        # Apply the mask to the original image to keep only the segment area
        color_segment = np.zeros_like(image_rgb)
        color_segment[segment] = image_rgb[segment]
        
        # Save the colored segment as a separate image
        cv2.imwrite(segment_filename, cv2.cvtColor(color_segment, cv2.COLOR_RGB2BGR))
        # print(f"Saved: {segment_filename}")
    # print("Segmentation and saving complete!")

IMAGE_LIST = train_images
for IMAGE_PATH in tqdm(IMAGE_LIST):
    IMAGE = str(IMAGE_PATH)
    id_ = IMAGE.split('-')[1].split('/')[0]
    if int(id_ ) > 1396:
        try:
            get_segments(IMAGE_PATH)
        except Exception as e:
            print(e)
    else:
        continue
