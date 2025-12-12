import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PhoBERTFactCheck(nn.Module):
    """
    PhoBERT for Fact-Checking (3 classes: Support, Refute, NEI).
    Built on top of vinai/phobert-base.
    """
    
    def __init__(
        self, 
        pretrained_name: str = "vinai/phobert-base", 
        num_classes: int = 3,
        dropout_rate: float = 0.15,
        freeze_bert: bool = False,
        unfreeze_last_n_layers: int = -1 # -1: Train all, 4: Train last 4 layers
    ):
        super(PhoBERTFactCheck, self).__init__()
        
        print(f"🏗️ Initializing PhoBERTFactCheck (base: {pretrained_name})...")
        
        # Load Config & Model
        print(f"⏳ Loading config...")
        self.config = AutoConfig.from_pretrained(pretrained_name)
        print(f"⏳ Loading PhoBERT weights (this may take 1-3 min on first run)...")
        self.bert = AutoModel.from_pretrained(pretrained_name)
        print(f"✅ PhoBERT loaded successfully!")
        
        # Layer Freezing Strategy
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("❄️ BERT encoder completely frozen.")
        elif unfreeze_last_n_layers > 0:
            # Freeze all parameters first
            for param in self.bert.parameters():
                param.requires_grad = False
                
            # Unfreeze Pooler (if exists)
            if hasattr(self.bert, 'pooler') and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
            
            # Unfreeze last N encoder layers
            # Roberta/PhoBERT structure: embeddings -> encoder -> layer (list)
            layers = self.bert.encoder.layer
            total_layers = len(layers)
            start_layer = max(0, total_layers - unfreeze_last_n_layers)
            
            for i in range(start_layer, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
            
            print(f"❄️ Layer Freezing: Frozen first {start_layer} layers. Training last {unfreeze_last_n_layers} layers.")
        else:
            print("🔥 Training ALL BERT layers.")
            
        # Classifier Head
        # Architecture: CLS -> LayerNorm -> Dropout -> Linear -> GELU -> Dropout -> Linear
        self.hidden_size = self.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        print(f"✅ Model initialized with {num_classes} classes.")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        """
        # BERT Encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Pooling Strategy: CLS Token (Standard for Classification)
        cls_embedding = last_hidden_state[:, 0, :]
        
        # Classifier
        logits = self.classifier(cls_embedding)
        
        return logits
