import pandas as pd
import numpy as np
import torch
import os
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm


train_df = pd.read_csv('final_data/Train/train.csv')
test_df = pd.read_csv('final_data/Test/test.csv')
val_df = pd.read_csv('final_data/Val/val.csv')

# Initialize the processor and model and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)

def get_embedding(image_path):
    try:
        # Load and ensure image is in RGB format
        image = Image.open(image_path).convert("RGB")
        
        # Get the bounding box of content, then crop
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        
        # Process the image using the model's processor
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy()
    except:
        return torch.zeros((1, 768))


# Function to load and sort segments in ascending order
def load_segments(meme_path):
    segments = sorted([os.path.join(meme_path, f) for f in os.listdir(meme_path) if f.endswith(".png")],
                      key=lambda x: int(os.path.basename(x).split('-')[-1].split('.')[0]))
    return {os.path.basename(segment): get_embedding(segment) for segment in segments}

base_dir = "sam/total_data/test"
memes = [os.path.join(base_dir, meme) for meme in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, meme))]
embeddings_dict = {os.path.basename(meme): load_segments(meme) for meme in tqdm(memes)}

base_dir = "sam/total_data/train"
train_memes = [os.path.join(base_dir, meme) for meme in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, meme))]
train_embeddings_dict = {os.path.basename(meme): load_segments(meme) for meme in tqdm(train_memes)}

topks = []
for current in tqdm(test_df['Post ID']):
    reference_meme_name = current
    reference_meme_embeddings = embeddings_dict[reference_meme_name]
    reference_values = np.vstack(list(reference_meme_embeddings.values()))

    scores = []
    try:
        for other_meme_name, other_meme_embeddings in train_embeddings_dict.items():
            other_values = np.vstack(list(other_meme_embeddings.values()))
            similarity_matrix = cosine_similarity(reference_values, other_values)
            score = np.sum(similarity_matrix)
            scores.append({
                'name':f'{current} vs {other_meme_name}',
                'score':score
            })
    except:
        scores.append(None)
    top_k1 = sorted(scores, key=lambda x: x['score'], reverse=True)
    topks.append(top_k1)
    
test_df['retrived'] = topks
test_df.to_csv('test_after_rag.csv', index=False)

topks = []
for current in tqdm(train_df['Post ID']):
    reference_meme_name = current
    reference_meme_embeddings = train_embeddings_dict[reference_meme_name]
    reference_values = np.vstack(list(reference_meme_embeddings.values()))

    scores = []
    try:
        for other_meme_name, other_meme_embeddings in train_embeddings_dict.items():
            other_values = np.vstack(list(other_meme_embeddings.values()))
            similarity_matrix = cosine_similarity(reference_values, other_values)
            score = np.sum(similarity_matrix)
            scores.append({
                'name':f'{current} vs {other_meme_name}',
                'score':score
            })
    except:
        scores.append(None)
    top_k1 = sorted(scores, key=lambda x: x['score'], reverse=True)
    topks.append(top_k1)
    
train_df['retrived'] = topks
train_df.to_csv('train_after_rag.csv', index=False)