import pandas as pd
import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np



def __main__():
    
    train_df = pd.read_csv('rag/train_raged_odi_3072_n1.csv')
    test_df = pd.read_csv('rag/test_raged_odi_3072_n1.csv')
    train_images = []
    for id in tqdm.tqdm(train_df['Post ID']):
        train_images.append([Image.open(f"sam/total_data/train/{id}/segment-{x+1}.png") for x in range(0, 5)])

    test_images = []
    for id in tqdm.tqdm(test_df['Post ID']):
        test_images.append([Image.open(f"sam/total_data/test/{id}/segment-{x+1}.png") for x in range(0, 5)])

    le = LabelEncoder()
    train_df['labels'] = le.fit_transform(train_df['Intensity'])
    test_df['labels'] = le.transform(test_df['Intensity'])
    classes = list(le.classes_)

    class_labels = train_df['labels'].values
    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    train_df['OCR'] = [x.replace('\n', ' ') for x in train_df['OCR']]
    test_df['OCR'] = [x.replace('\n', ' ') for x in test_df['OCR']]

    def prompt_it_decoder(df, split='train'):
        df['prompt'] = None
        prompts = []
        for n, (x,y, indices) in enumerate(zip(df['OCR'], df['Description'], df['topk_odi'])):
            if split=='train':
                indices = eval(indices)[1:]
            else:
                indices = eval(indices)
            idx_1, idx_2, idx_3 = indices[0],indices[1],indices[2]
            
            x_1, y_1, label_1 = train_df['OCR'][idx_1], train_df['Description'][idx_1], train_df['labels'][idx_1]
            x_2, y_2, label_2 = train_df['OCR'][idx_2], train_df['Description'][idx_2], train_df['labels'][idx_2]
            x_3, y_3, label_3 = train_df['OCR'][idx_3], train_df['Description'][idx_3], train_df['labels'][idx_3]
            prompt = f'''<|system|> You are tasked to Classify the intensity of the post using the following options: ["0 (Just a joke, not dark humor.)", "1 (Dark humor, but still generally inoffensive.)", "2 (Mildly dark or edgy, may be uncomfortable for some.)", "3 (Moderately dark, potentially offensive to sensitive individuals.)", "4 (Very dark humor, can hurt some social groups.)", "5 (Extremely offensive, can deeply hurt someone.)"].

    Example-1:
    <|OCR|>{x_1}\n<|Assistant|>The intensity of the post is: {label_1}.

    Example-2:
    <|OCR|>{x_2}\n<|Assistant|>The intensity of the post is: {label_2}.

    Example-3:
    <|OCR|>{x_3}\n<|Assistant|>The intensity of the post is: {label_3}.

    Your Turn:
    <|OCR|>{x}\n<|Assistant|>The intensity of the post is: '''
            
            prompts.append(prompt)
        
        df['prompt'] = prompts

        return df
    class TrustDataset(Dataset):
        def __init__(self, df):
            self.texts = df['prompt']
            self.labels = df['labels']
            self.ocrs = df['OCR']

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return {
                'prompt': self.texts[idx],
                'ocr': self.ocrs[idx],
                'targets': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    ### execution !
    batch_size = 16
    train_df = prompt_it_decoder(train_df, 'train')
    test_df = prompt_it_decoder(test_df, 'test')
    
    train_dataset = TrustDataset(train_df)
    val_dataset = TrustDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return  train_loader, val_loader, train_images, test_images,class_weights_tensor, batch_size, classes