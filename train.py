import torch.nn.functional as F
import tqdm
from data_loader import __main__
from model import *
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error
import os
from collections import defaultdict
import torch.optim as optim
import seaborn as sns
import pandas as pd

def plot_loss(loss_array, title="Loss Curve", xlabel="Steps", ylabel="Loss"):
    epochs = range(1, len(loss_array) + 1)  # X-axis from 1 to len(loss_array)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_array, marker='o', color='b', label='Loss')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = 'cuda'

# Define training and evaluation functions
def train_model(training_loader, model, optimizer, criterion, batch_size):
    model.train()
    train_losses = []
    loop = tqdm.tqdm(enumerate(training_loader), total=len(training_loader), leave=True)
    for batch_idx, data in loop:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            images = train_images[start_idx:end_idx]
            
            prompt, ocr, targets = data['prompt'], data['ocr'], data['targets'].to(device)
            optimizer.zero_grad()
            outputs = model(prompt, ocr, images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
    return np.mean(train_losses)

def eval_model(validation_loader, model, criterion):
    model.eval()
    val_targets, val_outputs, val_losses = [], [], []
    with torch.no_grad():
        loop = tqdm.tqdm(enumerate(validation_loader), total=len(validation_loader), leave=True)
        for batch_idx, data in loop:
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            images = test_images[start_idx:end_idx]
            
            prompt, ocr, targets = data['prompt'], data['ocr'], data['targets'].to(device)
            outputs = model(prompt, ocr, images)
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
            val_targets.extend(targets.cpu().numpy())
            val_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    accuracy = accuracy_score(val_targets, val_outputs)
    report = classification_report(val_targets, val_outputs, output_dict=True)
    mse = mean_squared_error(val_targets, val_outputs)
    return report, val_targets, val_outputs, np.mean(val_losses), accuracy, mse
# =================================================================================================================================
train_loader, val_loader, train_images, test_images, class_weights_tensor, batch_size, classes = __main__()
model = MultiModalClassifier("facebook/bart-large", "roberta-base", "openai/clip-vit-base-patch32", num_classes=8, pooler_dropout=0.2)

EPOCHS = 20
best_accuracy = 0
SAVE_DIR = "best_model"
os.makedirs(SAVE_DIR, exist_ok=True)
history = defaultdict(list)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss(class_weights_tensor.to(device))
criterion = nn.CrossEntropyLoss()
label_map = [0, 1, 2, 3, 4, 5]

for epoch in range(1, EPOCHS + 1):
    print(f'Epoch {epoch}/{EPOCHS}')
    train_loss = train_model(train_loader, model, optimizer, criterion, batch_size)
    report, val_targets, val_outputs_binary, val_loss, accuracy, val_mse = eval_model(val_loader, model, criterion)
    
    val_f1_macro = report['macro avg']['f1-score']
    val_f1_weighted = report['weighted avg']['f1-score']
    val_precision_macro = report['macro avg']['precision']
    val_recall_macro = report['macro avg']['recall']
    val_precision_weighted = report['weighted avg']['precision']
    val_recall_weighted = report['weighted avg']['recall']

    print(f"\n||Validation Accuracy: {accuracy:.4f} | Validation F1 Macro: {val_f1_macro:.4f} | Validation F1 Weighted: {val_f1_weighted:.4f}|")
    print(f"\n||Validation Precision Macro: {val_precision_macro:.4f} | Validation Recall Macro: {val_recall_macro:.4f}|")
    print(f"\n||Validation Precision Weighted: {val_precision_weighted:.4f} | Validation Recall Weighted: {val_recall_weighted:.4f}|")
    print(f"\n||Cross-entropy Loss: {val_loss:.4f}| MSE: {val_mse:.4f}|")  # Print MSE alongside cross-entropy loss
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1_weighted'].append(val_f1_weighted)
    
    if val_f1_macro > best_accuracy:
        best_accuracy = val_f1_macro
        
        model_path = os.path.join(SAVE_DIR, "best_model_intensity.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved the best model at epoch {epoch} to {model_path}")
        
        # Save the predictions
        predictions_df = pd.DataFrame({
            "Target": val_targets,
            "Prediction": val_outputs_binary
        })
        predictions_path = os.path.join(SAVE_DIR, "best_predictions_intensity.csv")
        predictions_df.to_csv(predictions_path, index=False)
        # Manually extract metrics from the classification report
        
        print("Classification Report:")
        print(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<15}")
        print("-" * 70)

        for key in sorted(report.keys()):
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = report[key]
                print(f"{label_map[int(key)]:<10}{metrics['precision']:<15.4f}{metrics['recall']:<15.4f}{metrics['f1-score']:<15.4f}{metrics['support']:<15.4f}")

        # Print the average metrics
        print(f"{'Accuracy':<10}{report['accuracy']:<15.4f}")
        print(f"{'Macro Avg':<10}{report['macro avg']['precision']:<15.4f}{report['macro avg']['recall']:<15.4f}{report['macro avg']['f1-score']:<15.4f}{report['macro avg']['support']:<15.4f}")
        print(f"{'Weighted Avg':<10}{report['weighted avg']['precision']:<15.4f}{report['weighted avg']['recall']:<15.4f}{report['weighted avg']['f1-score']:<15.4f}{report['weighted avg']['support']:<15.4f}\n\n\n")

# Plotting results
fig, axs = plt.subplots(2, 1, figsize=(5, 7))

# Plotting the training and validation losses
axs[0].plot(history['train_loss'], label='Train Loss')
axs[0].plot(history['val_loss'], label='Validation Loss')
axs[0].set_title('Training and Validation Losses')
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend()
axs[0].grid()

# Plotting the validation F1 scores
axs[1].plot(history['val_f1_macro'], label='Validation F1 Macro')
axs[1].plot(history['val_f1_weighted'], label='Validation F1 weighted')
axs[1].set_title('Validation F1 Scores')
axs[1].set_ylabel('F1 Score')
axs[1].set_xlabel('Epoch')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig('training_history.png')