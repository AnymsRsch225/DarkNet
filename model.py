import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
import torch.nn.functional as F

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

# Define the MultiModal Classifier
class MultiModalClassifier(nn.Module):
    # set num_classes according to the task, intensity cls and target cls differs!
    def __init__(self, bart_model_name, roberta_model_name, clip_model_name, num_classes=6, pooler_dropout=0.2, device='cuda'):
        super(MultiModalClassifier, self).__init__()
        self.device = device
        self.bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
        self.bart_model = AutoModel.from_pretrained(bart_model_name)
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        self.roberta_model = AutoModel.from_pretrained(roberta_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Projection layers and attention mechanism
        self.roberta_proj = nn.Linear(768, 1024)
        self.clip_proj = nn.Linear(512, 1024)
        self.mha = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        
        # Dropout and dense layers
        self.classification_head = BartClassificationHead(
            1024*2,
            1024,
            num_classes,
            pooler_dropout,
        )

    def forward(self, prompt, ocr_input, image_input):
        # Process BART input
        bart_inputs = self.bart_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        bart_outputs = self.bart_model(**bart_inputs.to(self.device)).last_hidden_state
        
        # Process RoBERTa input
        roberta_inputs = self.roberta_tokenizer(ocr_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
        roberta_outputs = self.roberta_model(**roberta_inputs.to(self.device)).last_hidden_state

        # Process CLIP input
        clip_outputs = [self.clip_model.get_image_features(**self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)).unsqueeze(0) 
                        for images in image_input]
        
        # Project the outputs to a common space
        clip_proj_out = self.clip_proj(torch.cat(clip_outputs))
        roberta_proj_out = self.roberta_proj(roberta_outputs)
        
        padding_needed = roberta_proj_out.size(1) - clip_proj_out.size(1)

        # Pad clip_output along the sequence length dimension
        if padding_needed > 0:
            clip_output_padded = F.pad(clip_proj_out, (0, 0, 0, padding_needed))
        else:
            clip_output_padded = clip_proj_out  # No padding if not needed
        attn_output, _ = self.mha(clip_output_padded, roberta_proj_out, clip_output_padded)
        mean_rep = attn_output.mean(dim=1)  # Mean pooling across sequence length
        #=======================================================================================================================
        eos_mask = bart_inputs['input_ids'].eq(self.bart_tokenizer.eos_token_id).to(bart_outputs.device)
        
        # Ensure all examples have the same number of <eos> tokens
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        
        # Extract hidden states corresponding to the EOS token
        sentence_representation = bart_outputs[eos_mask, :].view(bart_outputs.size(0), -1, bart_outputs.size(-1))[:, -1, :]
        
        # final_representation = torch.cat([sentence_representation, mean_rep], dim=1)
        final_representation = torch.cat([sentence_representation, mean_rep], dim=1)

        # Pass the concatenated representation through the classification head
        logits = self.classification_head(final_representation)

        return logits
