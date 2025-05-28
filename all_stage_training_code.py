#!/usr/bin/env python3
#=====================================================================
# ENHANCED SIGN LANGUAGE RECOGNITION - OPTIMIZED FOR BETTER PERFORMANCE
#=====================================================================

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
import math

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from collections import defaultdict
import editdistance
import json
import pickle

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_collate_fn(batch):
    """Enhanced collate function with better error handling"""
    frames_list = []
    gloss_texts = []
    folder_names = []
    gloss_sequences = []
    
    for item in batch:
        if item is not None:
            frames_list.append(item['frames'])
            gloss_texts.append(item['gloss_text'])
            folder_names.append(item['folder_name'])
            gloss_sequences.append(item['gloss_sequence'])
    
    if not frames_list:
        return None
    
    # Enhanced frame stacking with padding
    try:
        frames_batch = torch.stack(frames_list)
    except RuntimeError as e:
        print(f"Error stacking frames: {e}")
        max_frames = max(f.shape[0] for f in frames_list)
        padded_frames = []
        for frames in frames_list:
            if frames.shape[0] < max_frames:
                padding_shape = (max_frames - frames.shape[0],) + frames.shape[1:]
                padding = torch.zeros(padding_shape, dtype=frames.dtype)
                padded_frames.append(torch.cat([frames, padding], dim=0))
            else:
                padded_frames.append(frames)
        frames_batch = torch.stack(padded_frames)
    
    return {
        'frames': frames_batch,
        'gloss_text': gloss_texts,
        'folder_name': folder_names,
        'gloss_sequence': gloss_sequences
    }

class EnhancedSignLanguageDataset(Dataset):
    """Enhanced dataset with improved data augmentation and preprocessing"""
    
    def __init__(self, root_train_folder, annotations_folder, transform=None, max_frames=20, 
                 is_training=True, augment_prob=0.7):
        self.root_train_folder = root_train_folder
        self.transform = transform
        self.max_frames = max_frames
        self.is_training = is_training
        self.augment_prob = augment_prob
        
        # Load annotations
        csv_files = [f for f in os.listdir(annotations_folder) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV file found in annotations folder")
        
        csv_path = os.path.join(annotations_folder, csv_files[0])
        self.annotations = pd.read_csv(csv_path)
        self.annotations = self.annotations.iloc[1:, :2]
        self.annotations.columns = ['folder_name', 'gloss_text']
        
        # Enhanced vocabulary creation
        self.create_enhanced_vocabulary()
        
        # Filter annotations
        self.valid_annotations = []
        for _, row in self.annotations.iterrows():
            folder_path = os.path.join(root_train_folder, str(row['folder_name']))
            if os.path.exists(folder_path):
                gloss_sequence = self.text_to_sequence(str(row['gloss_text']))
                if len(gloss_sequence) > 0:  # Only include non-empty sequences
                    self.valid_annotations.append({
                        'folder_name': str(row['folder_name']),
                        'gloss_text': str(row['gloss_text']).lower().strip(),
                        'folder_path': folder_path,
                        'gloss_sequence': gloss_sequence
                    })
        
        # Enhanced data augmentation
        self.setup_augmentations()
        
        print(f"📊 Dataset Statistics:")
        print(f"   Valid samples: {len(self.valid_annotations)}")
        print(f"   Vocabulary size: {len(self.vocab)}")
        print(f"   Max frames: {max_frames}")
        print(f"   Training mode: {is_training}")
        print(f"   Augmentation probability: {augment_prob}")
        
        # Save vocabulary for inference
        self.save_vocabulary()
    
    def create_enhanced_vocabulary(self):
        """Create enhanced vocabulary with frequency-based ordering"""
        # Collect all words with frequencies
        word_freq = defaultdict(int)
        for _, row in self.annotations.iterrows():
            gloss_text = str(row['gloss_text']).lower().strip()
            words = gloss_text.split()
            for word in words:
                if word and word.isalpha():  # Filter out empty and non-alphabetic
                    word_freq[word] += 1
        
        # Sort by frequency (most common first)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Create vocabulary with special tokens
        self.vocab = {
            '<blank>': 0,  # CTC blank token
            '<unk>': 1,    # Unknown token
            '<pad>': 2,    # Padding token
        }
        
        # Add words in frequency order
        for word, freq in sorted_words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.idx_to_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"📚 Vocabulary created with {len(self.vocab)} tokens")
        print(f"   Most frequent words: {[word for word, _ in sorted_words[:10]]}")
        
    def save_vocabulary(self):
        """Save vocabulary for inference"""
        vocab_path = os.path.join(os.path.dirname(self.root_train_folder), 'vocabulary.pkl')
        try:
            with open(vocab_path, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'idx_to_vocab': self.idx_to_vocab
                }, f)
            print(f"📝 Vocabulary saved to: {vocab_path}")
        except Exception as e:
            print(f"⚠️  Could not save vocabulary: {e}")
    
    def setup_augmentations(self):
        """Setup enhanced data augmentations for sign language"""
        if self.is_training:
            self.spatial_augment = A.Compose([
                A.HorizontalFlip(p=0.5),  # Mirror for sign language
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.OneOf([
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                ], p=0.3),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            ], p=self.augment_prob)
        else:
            self.spatial_augment = None
    
    def text_to_sequence(self, text):
        """Enhanced text to sequence conversion"""
        words = text.lower().strip().split()
        sequence = []
        for word in words:
            word = word.strip()
            if word:  # Skip empty words
                if word in self.vocab:
                    sequence.append(self.vocab[word])
                else:
                    sequence.append(self.vocab['<unk>'])
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence back to text"""
        words = []
        for idx in sequence:
            if idx in self.idx_to_vocab and idx not in [0, 2]:  # Skip blank and pad
                words.append(self.idx_to_vocab[idx])
        return ' '.join(words)
    
    def __len__(self):
        return len(self.valid_annotations)
    
    def load_frames_enhanced(self, folder_path):
        """Enhanced frame loading with better sampling"""
        frame_files = sorted([f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(frame_files) == 0:
            raise ValueError(f"No image files found in {folder_path}")
        
        # Intelligent frame sampling
        if len(frame_files) > self.max_frames:
            if self.is_training:
                # Random sampling for training (data augmentation)
                start_idx = np.random.randint(0, max(1, len(frame_files) - self.max_frames))
                end_idx = start_idx + self.max_frames
                frame_files = frame_files[start_idx:end_idx]
            else:
                # Uniform sampling for validation/test
                indices = np.linspace(0, len(frame_files)-1, self.max_frames, dtype=int)
                frame_files = [frame_files[i] for i in indices]
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame)
            except Exception as e:
                continue
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames loaded from {folder_path}")
        
        # Temporal augmentation - frame repetition
        while len(frames) < self.max_frames:
            if self.is_training:
                # Random frame repetition
                frames.append(frames[np.random.randint(0, len(frames))].copy())
            else:
                # Last frame repetition
                frames.append(frames[-1].copy())
        
        frames = frames[:self.max_frames]
        return np.stack(frames)
    
    def __getitem__(self, idx):
        """Enhanced getitem with augmentations"""
        if idx >= len(self.valid_annotations):
            idx = idx % len(self.valid_annotations)
            
        sample = self.valid_annotations[idx]
        
        try:
            # Load frames
            frames = self.load_frames_enhanced(sample['folder_path'])
            
            # Apply spatial augmentations
            if self.spatial_augment and self.is_training:
                augmented_frames = []
                for frame in frames:
                    augmented = self.spatial_augment(image=frame)['image']
                    augmented_frames.append(augmented)
                frames = np.stack(augmented_frames)
            
            # Apply transforms
            if self.transform:
                transformed_frames = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        frame_pil = Image.fromarray(frame.astype(np.uint8))
                    else:
                        frame_pil = frame
                    transformed_frame = self.transform(frame_pil)
                    transformed_frames.append(transformed_frame)
                frames = torch.stack(transformed_frames)
            else:
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            
            return {
                'frames': frames,
                'gloss_text': sample['gloss_text'],
                'folder_name': sample['folder_name'],
                'gloss_sequence': sample['gloss_sequence']
            }
            
        except Exception as e:
            print(f"Error processing sample {idx} ({sample['folder_name']}): {e}")
            # Return dummy data
            dummy_frames = torch.zeros(self.max_frames, 3, 112, 112)
            return {
                'frames': dummy_frames,
                'gloss_text': "unknown",
                'folder_name': "error",
                'gloss_sequence': [1]  # <unk> token
            }

def get_enhanced_data_loaders(root_train_folder, annotations_folder, 
                             batch_size=4, test_split=0.2, max_frames=20, num_workers=2):
    """Enhanced data loaders with better preprocessing"""
    from sklearn.model_selection import train_test_split
    
    # Enhanced transforms with normalization
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = EnhancedSignLanguageDataset(
        root_train_folder=root_train_folder,
        annotations_folder=annotations_folder,
        transform=train_transform,
        max_frames=max_frames,
        is_training=True
    )
    
    # Split data
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_split, random_state=42, shuffle=True
    )
    
    # Create training dataset
    train_dataset = EnhancedSignLanguageDataset(
        root_train_folder=root_train_folder,
        annotations_folder=annotations_folder,
        transform=train_transform,
        max_frames=max_frames,
        is_training=True
    )
    train_dataset.valid_annotations = [full_dataset.valid_annotations[i] for i in train_indices]
    train_dataset.vocab = full_dataset.vocab
    train_dataset.idx_to_vocab = full_dataset.idx_to_vocab
    
    # Create validation dataset
    val_dataset = EnhancedSignLanguageDataset(
        root_train_folder=root_train_folder,
        annotations_folder=annotations_folder,
        transform=val_transform,
        max_frames=max_frames,
        is_training=False
    )
    val_dataset.valid_annotations = [full_dataset.valid_annotations[i] for i in val_indices]
    val_dataset.vocab = full_dataset.vocab
    val_dataset.idx_to_vocab = full_dataset.idx_to_vocab
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader, full_dataset

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for better feature focus"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class EnhancedConvGRU(nn.Module):
    """Enhanced ConvGRU with attention mechanism"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(EnhancedConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Standard ConvGRU gates
        self.conv_reset = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_update = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_new = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
        # Attention mechanism
        self.spatial_attention = SpatialAttention(hidden_dim)
        
        # Normalization
        self.norm = nn.GroupNorm(8, hidden_dim)
    
    def forward(self, input_tensor, hidden_state):
        if hidden_state is None:
            batch_size, _, height, width = input_tensor.size()
            hidden_state = torch.zeros(batch_size, self.hidden_dim, height, width, 
                                     device=input_tensor.device)
        
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        
        reset_gate = torch.sigmoid(self.conv_reset(combined))
        update_gate = torch.sigmoid(self.conv_update(combined))
        
        reset_hidden = reset_gate * hidden_state
        combined_reset = torch.cat([input_tensor, reset_hidden], dim=1)
        new_gate = torch.tanh(self.conv_new(combined_reset))
        
        new_hidden = (1 - update_gate) * new_gate + update_gate * hidden_state
        
        # Apply attention and normalization
        new_hidden = self.spatial_attention(new_hidden)
        new_hidden = self.norm(new_hidden)
        
        return new_hidden

class EnhancedFourStageModel(nn.Module):
    """Enhanced 4-stage model with 1024D features and modern optimizations"""
    
    def __init__(self, vocab_size, feature_dim=1024, hidden_dim=512, lstm_hidden=256, 
                 num_heads=8, dropout=0.1):
        super(EnhancedFourStageModel, self).__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        
        print(f"🏗️  Initializing Enhanced 4-Stage Model:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Feature dim: {feature_dim}")  # 1024
        print(f"   Hidden dim: {hidden_dim}")    # 512
        print(f"   LSTM hidden: {lstm_hidden}")  # 256
        print(f"   Attention heads: {num_heads}")
        
        # Stage 1: Enhanced Visual Feature Extraction
        self.visual_encoder = self._build_enhanced_visual_encoder()
        self.text_encoder = self._build_enhanced_text_encoder()
        
        # Stage 2: Enhanced Motion-aware Keyframe Selection
        self.conv_gru = EnhancedConvGRU(input_dim=feature_dim, hidden_dim=hidden_dim)
        
        # Multi-scale temporal pooling
        self.temporal_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
        ])
        
        self.keyframe_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # *2 for avg+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Stage 3: Enhanced Temporal Modeling
        self.temporal_lstm = nn.LSTM(
            input_size=feature_dim * 4 * 4,  # 1024 * 4 * 4
            hidden_size=lstm_hidden,
            num_layers=3,  # Increased layers
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_hidden > 1 else 0
        )
        
        # Multi-head attention for temporal modeling
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced frame classifier
        self.frame_classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, vocab_size)
        )
        
        # Stage 4: Enhanced CTC Decoder
        self.ctc_projection = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, vocab_size)
        )
        
        # Learnable loss weights
        self.loss_weights = nn.Parameter(torch.ones(5))  # 5 losses
        
        # Initialize weights
        self._initialize_weights()
        
        print("✅ Enhanced model architecture initialized successfully")
    
    def _build_enhanced_visual_encoder(self):
        """Enhanced visual encoder with 1024D features"""
        print("🏗️  Building enhanced visual encoder (ResNet50 -> 1024D)...")
        
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        layers = list(resnet.children())[:-2]
        visual_encoder = nn.Sequential(*layers)
        
        # Enhanced projection with 1024D output
        visual_encoder.add_module('projection', nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(2048, self.feature_dim, 1),  # 2048 -> 1024
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU()
        ))
        
        return visual_encoder
    
    def _build_enhanced_text_encoder(self):
        """Enhanced text encoder with 1024D features"""
        print("🏗️  Building enhanced text encoder (BERT -> 1024D)...")
        
        class EnhancedTextEncoder(nn.Module):
            def __init__(self, feature_dim=1024):
                super().__init__()
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                
                # Freeze early layers, unfreeze later layers
                for i, layer in enumerate(self.bert.encoder.layer):
                    if i < 8:  # Freeze first 8 layers
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:  # Unfreeze last 4 layers
                        for param in layer.parameters():
                            param.requires_grad = True
                
                # Enhanced projection to 1024D
                self.projection = nn.Sequential(
                    nn.Linear(768, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.ReLU()
                )
            
            def forward(self, text_list):
                encoded = self.tokenizer(
                    text_list,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt',
                    return_attention_mask=True
                )
                
                device = next(self.parameters()).device
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use pooled output + CLS token
                pooled_output = outputs.pooler_output
                cls_output = outputs.last_hidden_state[:, 0, :]
                
                # Combine both
                combined = torch.cat([pooled_output, cls_output], dim=-1)
                combined = nn.Linear(1536, 768).to(device)(combined)
                
                return self.projection(combined)
        
        return EnhancedTextEncoder(self.feature_dim)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def stage1_enhanced_visual_extraction(self, frames):
        """Enhanced Stage 1 - Visual feature extraction"""
        B, T, C, H, W = frames.shape
        
        # Process all frames
        frames_flat = frames.view(B * T, C, H, W)
        visual_features = self.visual_encoder(frames_flat)  # (B*T, 1024, 4, 4)
        
        # Reshape back
        _, feat_dim, feat_h, feat_w = visual_features.shape
        visual_features = visual_features.view(B, T, feat_dim, feat_h, feat_w)
        
        return visual_features
    
    def stage2_enhanced_keyframe_selection(self, visual_features, k_ratio=0.8):
        """Enhanced Stage 2 - Keyframe selection with attention"""
        B, T, C, H, W = visual_features.shape
        
        # Enhanced motion computation
        frame_diffs = visual_features[:, 1:] - visual_features[:, :-1]
        
        # Apply ConvGRU
        hidden_state = None
        gru_outputs = []
        
        for t in range(T - 1):
            hidden_state = self.conv_gru(frame_diffs[:, t], hidden_state)
            gru_outputs.append(hidden_state)
        
        gru_features = torch.stack(gru_outputs, dim=1)  # (B, T-1, hidden_dim, H, W)
        
        # Multi-scale pooling
        pooled_features = []
        for pool in self.temporal_pools:
            pooled = pool(gru_features.view(B * (T-1), -1, H, W))
            pooled = pooled.view(B, T-1, -1)
            pooled_features.append(pooled)
        
        combined_features = torch.cat(pooled_features, dim=-1)
        
        # Compute importance scores
        frame_scores = self.keyframe_selector(combined_features)  # (B, T-1, 1)
        frame_scores = frame_scores.squeeze(-1)  # (B, T-1)
        frame_scores = F.softmax(frame_scores, dim=1)
        
        # Select top-k frames
        k = max(1, int((T-1) * k_ratio))
        _, top_indices = torch.topk(frame_scores, k, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)
        
        # Select keyframes
        selected_features = []
        for b in range(B):
            batch_features = visual_features[b, 1:][top_indices[b]]
            selected_features.append(batch_features)
        
        selected_features = torch.stack(selected_features)
        
        return selected_features, frame_scores, top_indices
    
    def stage3_enhanced_temporal_modeling(self, selected_features):
        """Enhanced Stage 3 - Temporal modeling with attention"""
        B, K, C, H, W = selected_features.shape
        
        # Flatten spatial dimensions
        features_flat = selected_features.view(B, K, C * H * W)
        
        # Bi-LSTM processing
        lstm_out, (hidden, cell) = self.temporal_lstm(features_flat)
        
        # Apply temporal attention
        attended_features, attention_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Combine LSTM and attention features
        combined_features = lstm_out + attended_features
        
        # Frame-wise classification
        frame_logits = self.frame_classifier(combined_features)
        
        return combined_features, frame_logits, attention_weights
    
    def stage4_enhanced_ctc_prediction(self, lstm_features):
        """Enhanced Stage 4 - CTC prediction"""
        ctc_logits = self.ctc_projection(lstm_features)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        return ctc_log_probs
    
    def forward(self, frames, gloss_texts=None, gloss_sequences=None, mode='train'):
        """Enhanced forward pass"""
        outputs = {}
        
        # Stage 1: Enhanced Visual Feature Extraction
        visual_features = self.stage1_enhanced_visual_extraction(frames)
        outputs['visual_features'] = visual_features
        
        # Text encoding
        if gloss_texts is not None:
            try:
                text_features = self.text_encoder(gloss_texts)
                outputs['text_features'] = text_features
            except Exception as e:
                print(f"⚠️  Text encoding error: {e}")
        
        # Stage 2: Enhanced Keyframe Selection
        selected_features, frame_scores, keyframe_indices = self.stage2_enhanced_keyframe_selection(visual_features)
        outputs['selected_features'] = selected_features
        outputs['frame_scores'] = frame_scores
        outputs['keyframe_indices'] = keyframe_indices
        
        # Stage 3: Enhanced Temporal Modeling
        lstm_features, frame_logits, attention_weights = self.stage3_enhanced_temporal_modeling(selected_features)
        outputs['lstm_features'] = lstm_features
        outputs['frame_logits'] = frame_logits
        outputs['attention_weights'] = attention_weights
        
        # Stage 4: Enhanced CTC Prediction
        ctc_logits = self.stage4_enhanced_ctc_prediction(lstm_features)
        outputs['ctc_logits'] = ctc_logits
        
        return outputs

class EnhancedMultiStageLoss(nn.Module):
    """Enhanced multi-stage loss with label smoothing and focal loss"""
    
    def __init__(self, vocab_size, label_smoothing=0.1, focal_alpha=1.0, focal_gamma=2.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        
        # Enhanced losses
        self.cross_modal_loss = nn.CosineEmbeddingLoss()
        self.classification_loss = nn.CrossEntropyLoss(
            ignore_index=-1, 
            label_smoothing=label_smoothing
        )
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.focal_loss = self._focal_loss
        
        # Attention regularization
        self.attention_regularization = self._attention_regularization
        
    def _focal_loss(self, inputs, targets, alpha=1.0, gamma=2.0):
        """Focal loss for hard example mining"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
    def _attention_regularization(self, attention_weights):
        """Regularize attention to be smooth"""
        if attention_weights is None:
            return torch.tensor(0.0)
        
        # Encourage attention diversity
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        return -entropy.mean()  # Negative because we want high entropy
    
    def forward(self, outputs, targets):
        """Enhanced multi-stage loss computation"""
        losses = {}
        device = next(iter(outputs.values())).device
        
        # 1. Cross-modal alignment loss
        if 'text_features' in outputs and 'visual_features' in outputs:
            try:
                visual_pooled = outputs['visual_features'].mean(dim=(1, 3, 4))  # Pool over time and space
                text_features = outputs['text_features']
                
                # Normalize features
                visual_pooled = F.normalize(visual_pooled, dim=1)
                text_features = F.normalize(text_features, dim=1)
                
                batch_size = visual_pooled.size(0)
                labels = torch.ones(batch_size, device=device)
                losses['cross_modal'] = self.cross_modal_loss(visual_pooled, text_features, labels)
            except Exception as e:
                losses['cross_modal'] = torch.tensor(0.0, requires_grad=True, device=device)
        
        # 2. Keyframe selection loss (entropy regularization)
        if 'frame_scores' in outputs:
            frame_scores = outputs['frame_scores']
            entropy = -torch.sum(frame_scores * torch.log(frame_scores + 1e-8), dim=1).mean()
            losses['keyframe_selection'] = -entropy
        
        # 3. Enhanced frame classification loss
        if 'frame_logits' in outputs and 'frame_labels' in targets:
            try:
                frame_logits = outputs['frame_logits']
                frame_labels = targets['frame_labels']
                
                B, K, V = frame_logits.shape
                frame_logits_flat = frame_logits.view(-1, V)
                frame_labels_flat = frame_labels.view(-1)
                
                # Standard cross-entropy
                losses['frame_classification'] = self.classification_loss(frame_logits_flat, frame_labels_flat)
                
                # Focal loss for hard examples
                valid_mask = frame_labels_flat != -1
                if valid_mask.any():
                    losses['focal_frame'] = self.focal_loss(
                        frame_logits_flat[valid_mask], 
                        frame_labels_flat[valid_mask]
                    )
                else:
                    losses['focal_frame'] = torch.tensor(0.0, requires_grad=True, device=device)
                    
            except Exception as e:
                losses['frame_classification'] = torch.tensor(0.0, requires_grad=True, device=device)
                losses['focal_frame'] = torch.tensor(0.0, requires_grad=True, device=device)
        
        # 4. Enhanced CTC loss
        if 'ctc_logits' in outputs and 'gloss_sequences' in targets:
            try:
                ctc_logits = outputs['ctc_logits']
                gloss_sequences = targets['gloss_sequences']
                
                # Prepare for CTC loss
                ctc_logits = ctc_logits.permute(1, 0, 2)  # (T, B, V)
                
                input_lengths = torch.full((ctc_logits.size(1),), ctc_logits.size(0), dtype=torch.long)
                target_lengths = torch.tensor([len(seq) for seq in gloss_sequences], dtype=torch.long)
                targets_flat = torch.cat([torch.tensor(seq, dtype=torch.long) for seq in gloss_sequences])
                
                # Move to device
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                targets_flat = targets_flat.to(device)
                
                losses['ctc'] = self.ctc_loss(ctc_logits, targets_flat, input_lengths, target_lengths)
                
            except Exception as e:
                losses['ctc'] = torch.tensor(0.0, requires_grad=True, device=device)
        
        # 5. Attention regularization
        if 'attention_weights' in outputs:
            losses['attention_reg'] = self.attention_regularization(outputs['attention_weights'])
        
        return losses

class EnhancedTrainer:
    """Enhanced trainer with modern optimization techniques"""
    
    def __init__(self, model, train_loader, val_loader, vocab_size, device='cuda', 
                 save_dir='enhanced_checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Enhanced loss
        self.criterion = EnhancedMultiStageLoss(vocab_size, label_smoothing=0.1)
        
        # Enhanced optimizer with different learning rates
        param_groups = [
            {'params': self.model.visual_encoder.parameters(), 'lr': 1e-5, 'name': 'visual'},
            {'params': self.model.text_encoder.parameters(), 'lr': 5e-6, 'name': 'text'},
            {'params': self.model.conv_gru.parameters(), 'lr': 5e-4, 'name': 'gru'},
            {'params': self.model.temporal_lstm.parameters(), 'lr': 2e-4, 'name': 'lstm'},
            {'params': self.model.temporal_attention.parameters(), 'lr': 2e-4, 'name': 'attention'},
            {'params': self.model.frame_classifier.parameters(), 'lr': 2e-4, 'name': 'classifier'},
            {'params': self.model.ctc_projection.parameters(), 'lr': 2e-4, 'name': 'ctc'},
            {'params': [self.model.loss_weights], 'lr': 1e-3, 'name': 'weights'}
        ]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        
        # Enhanced scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        
        # Training metrics
        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Gradient accumulation
        self.accumulation_steps = 2
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        
        print("✅ Enhanced trainer initialized successfully")
        print(f"   Device: {device}")
        print(f"   Optimizer: AdamW with {len(param_groups)} parameter groups")
        print(f"   Scheduler: CosineAnnealingWarmRestarts")
        print(f"   Gradient accumulation: {self.accumulation_steps}")
        print(f"   Early stopping patience: {self.patience}")
    
    def prepare_enhanced_targets(self, batch):
        """Prepare enhanced targets for training"""
        targets = {}
        
        # Gloss sequences for CTC
        targets['gloss_sequences'] = batch['gloss_sequence']
        
        # Enhanced frame labels
        max_frames = 10  # Updated max frames
        k_ratio = 0.8    # Updated k_ratio
        actual_keyframes = max(1, int((max_frames - 1) * k_ratio))  # 15 keyframes
        
        frame_labels = []
        for seq in batch['gloss_sequence']:
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            
            if len(seq_tensor) <= actual_keyframes:
                # Pad with -1 (ignore index)
                padded = torch.cat([
                    seq_tensor, 
                    torch.full((actual_keyframes - len(seq_tensor),), -1)
                ])
            else:
                # Truncate and repeat last token
                padded = seq_tensor[:actual_keyframes]
            
            frame_labels.append(padded)
        
        targets['frame_labels'] = torch.stack(frame_labels).to(self.device)
        return targets
    
    def train_epoch(self, epoch):
        """Enhanced training epoch"""
        self.model.train()
        total_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        successful_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
                
            try:
                frames = batch['frames'].to(self.device)
                gloss_texts = batch['gloss_text']
                
                # Prepare targets
                targets = self.prepare_enhanced_targets(batch)
                
                # Forward pass
                outputs = self.model(frames, gloss_texts, batch['gloss_sequence'], mode='train')
                
                # Compute losses
                losses = self.criterion(outputs, targets)
                
                # Combine losses with learnable weights
                loss_weights = F.softmax(self.model.loss_weights, dim=0)
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                loss_count = 0
                for i, (loss_name, loss_value) in enumerate(losses.items()):
                    if i < len(loss_weights) and torch.isfinite(loss_value) and loss_value > 0:
                        weighted_loss = loss_weights[i] * loss_value
                        total_loss = total_loss + weighted_loss
                        total_losses[loss_name] += loss_value.item()
                        loss_count += 1
                
                if loss_count > 0 and torch.isfinite(total_loss):
                    # Gradient accumulation
                    total_loss = total_loss / self.accumulation_steps
                    total_loss.backward()
                    
                    total_losses['total'] += total_loss.item() * self.accumulation_steps
                    
                    # Update weights every accumulation_steps
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    successful_batches += 1
                
                # Update progress bar
                if successful_batches > 0:
                    avg_loss = total_losses['total'] / successful_batches
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'Success': f'{successful_batches}/{batch_idx+1}'
                    })
                
                # Memory cleanup
                del frames, outputs, losses, total_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n💥 CUDA OOM in batch {batch_idx}. Skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"\n❌ Runtime error: {e}")
                    continue
            except Exception as e:
                print(f"\n⚠️  Error in batch {batch_idx}: {e}")
                continue
        
        # Final optimizer step
        if successful_batches % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Average losses
        if successful_batches > 0:
            avg_losses = {k: v / successful_batches for k, v in total_losses.items()}
        else:
            avg_losses = {'total': float('inf')}
        
        # Store losses
        for k, v in avg_losses.items():
            self.train_losses[k].append(v)
        
        print(f"\n📊 Epoch {epoch+1} Training: {successful_batches}/{num_batches} successful batches")
        return avg_losses
    
    def validate(self, epoch):
        """Enhanced validation"""
        self.model.eval()
        total_losses = defaultdict(float)
        all_predictions = []
        all_ground_truths = []
        successful_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                if batch is None:
                    continue
                    
                try:
                    frames = batch['frames'].to(self.device)
                    gloss_texts = batch['gloss_text']
                    targets = self.prepare_enhanced_targets(batch)
                    
                    # Forward pass
                    outputs = self.model(frames, gloss_texts, batch['gloss_sequence'], mode='eval')
                    
                    # Compute losses
                    losses = self.criterion(outputs, targets)
                    
                    for loss_name, loss_value in losses.items():
                        if torch.isfinite(loss_value):
                            total_losses[loss_name] += loss_value.item()
                    
                    # Collect predictions for metrics
                    if 'ctc_logits' in outputs:
                        ctc_logits = outputs['ctc_logits']
                        predictions = torch.argmax(ctc_logits, dim=-1)
                        all_predictions.extend(predictions.cpu().numpy())
                        all_ground_truths.extend(batch['gloss_sequence'])
                    
                    successful_batches += 1
                    
                    # Memory cleanup
                    del frames, outputs, losses
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Validation error in batch {batch_idx}: {e}")
                    continue
        
        # Average losses
        if successful_batches > 0:
            avg_losses = {k: v / successful_batches for k, v in total_losses.items()}
            
            # Compute metrics
            if all_predictions and all_ground_truths:
                wer = self.compute_word_error_rate(all_predictions, all_ground_truths)
                seq_acc = self.compute_sequence_accuracy(all_predictions, all_ground_truths)
                avg_losses['wer'] = wer
                avg_losses['seq_acc'] = seq_acc
                
                print(f"📈 Validation - WER: {wer:.4f}, Seq Acc: {seq_acc:.4f}")
        else:
            avg_losses = {'total': float('inf'), 'wer': 1.0, 'seq_acc': 0.0}
        
        # Store losses
        for k, v in avg_losses.items():
            self.val_losses[k].append(v)
        
        print(f"✅ Validation completed: {successful_batches}/{len(self.val_loader)} successful batches")
        return avg_losses
    
    def compute_word_error_rate(self, predictions, ground_truths):
        """Compute Word Error Rate"""
        total_errors = 0
        total_words = 0
        
        for pred, truth in zip(predictions, ground_truths):
            # CTC decoding
            pred_clean = []
            prev = -1
            for p in pred:
                if p != 0 and p != prev:  # Remove blanks and duplicates
                    pred_clean.append(p)
                prev = p
            
            truth_list = truth if isinstance(truth, list) else [truth]
            total_words += len(truth_list)
            
            if len(truth_list) > 0:
                errors = editdistance.eval(pred_clean, truth_list)
                total_errors += errors
        
        return total_errors / max(total_words, 1)
    
    def compute_sequence_accuracy(self, predictions, ground_truths):
        """Compute sequence-level accuracy"""
        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            pred_clean = []
            prev = -1
            for p in pred:
                if p != 0 and p != prev:
                    pred_clean.append(p)
                prev = p
            
            if pred_clean == truth:
                correct += 1
        
        return correct / len(predictions)
    
    def save_training_visualizations(self, epoch):
        """Save training visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Epoch {epoch+1}', fontsize=16)
            
            # Training losses
            if self.train_losses['total']:
                axes[0, 0].plot(self.train_losses['total'], label='Total Loss', color='blue')
                if 'ctc' in self.train_losses and self.train_losses['ctc']:
                    axes[0, 0].plot(self.train_losses['ctc'], label='CTC Loss', color='red', alpha=0.7)
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Validation losses
            if self.val_losses['total']:
                axes[0, 1].plot(self.val_losses['total'], label='Total Loss', color='green')
                if 'wer' in self.val_losses and self.val_losses['wer']:
                    axes[0, 1].plot(self.val_losses['wer'], label='WER', color='orange', alpha=0.7)
                axes[0, 1].set_title('Validation Metrics')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Metric')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Learning rate
            lr_history = [group['lr'] for group in self.optimizer.param_groups]
            if len(lr_history) > 0:
                axes[1, 0].plot([lr_history[0]] * (epoch + 1), label='Visual LR', color='purple')
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Loss weights
            if hasattr(self.model, 'loss_weights'):
                weights = F.softmax(self.model.loss_weights, dim=0).cpu().detach().numpy()
                weight_names = ['Cross Modal', 'Keyframe', 'Frame Class', 'Focal', 'CTC'][:len(weights)]
                axes[1, 1].bar(weight_names, weights, color='skyblue')
                axes[1, 1].set_title('Loss Weights')
                axes[1, 1].set_ylabel('Weight')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'training_progress_epoch_{epoch+1}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not save training visualizations: {e}")
    
    def save_model(self, path, epoch, is_best=False):
        """Save enhanced model with complete state"""
        try:
            # Get vocabulary if available
            vocab = None
            if hasattr(self.train_loader.dataset, 'vocab'):
                vocab = self.train_loader.dataset.vocab
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_losses': dict(self.train_losses),
                'val_losses': dict(self.val_losses),
                'vocab_size': self.vocab_size,
                'vocab': vocab,
                'best_val_loss': self.best_val_loss,
                'model_config': {
                    'feature_dim': self.model.feature_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'lstm_hidden': self.model.lstm_hidden,
                    'num_heads': self.model.num_heads,
                    'dropout': self.model.dropout
                },
                'timestamp': datetime.now().isoformat(),
                'frame_size': '112x112',
                'enhanced_version': True
            }
            
            torch.save(save_dict, path)
            
            status = "BEST " if is_best else ""
            print(f"💾 {status}Model saved: {path}")
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def train(self, num_epochs):
        """Enhanced training loop"""
        print(f"🚀 Starting enhanced training for {num_epochs} epochs")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            try:
                epoch_start_time = datetime.now()
                
                # Training
                print(f"\n🔄 EPOCH {epoch+1}/{num_epochs}")
                train_losses = self.train_epoch(epoch)
                
                # Validation
                val_losses = self.validate(epoch)
                
                # Scheduler step
                self.scheduler.step()
                
                # Check for improvement
                current_val_loss = val_losses.get('total', float('inf'))
                is_best = current_val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = current_val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    best_path = os.path.join(self.save_dir, 'best_enhanced_model.pth')
                    self.save_model(best_path, epoch, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Save training visualizations
                if (epoch + 1) % 5 == 0:
                    self.save_training_visualizations(epoch)
                
                # Save checkpoint
                if (epoch + 1) % 20 == 0:
                    checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                    self.save_model(checkpoint_path, epoch)
                
                # Print epoch summary
                epoch_time = datetime.now() - epoch_start_time
                print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
                print(f"   Duration: {epoch_time}")
                print(f"   Train Loss: {train_losses.get('total', 0):.4f}")
                print(f"   Val Loss: {val_losses.get('total', 0):.4f}")
                print(f"   WER: {val_losses.get('wer', 1.0):.4f}")
                print(f"   Seq Acc: {val_losses.get('seq_acc', 0.0):.4f}")
                print(f"   Best Val Loss: {self.best_val_loss:.4f}")
                print(f"   Patience: {self.patience_counter}/{self.patience}")
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                    break
                
                print("-" * 70)
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Training interrupted at epoch {epoch+1}")
                interrupt_path = os.path.join(self.save_dir, f'interrupted_epoch_{epoch+1}.pth')
                self.save_model(interrupt_path, epoch)
                break
                
            except Exception as e:
                print(f"\n❌ Error in epoch {epoch+1}: {e}")
                continue
            # Log loss weights after each epoch
            if not hasattr(self, 'loss_weight_log'):
                self.loss_weight_log = []

            with torch.no_grad():
                softmax_weights = F.softmax(self.model.loss_weights, dim=0).cpu().numpy()
                self.loss_weight_log.append(softmax_weights.tolist())

        
        # Save final model
        final_path = os.path.join(self.save_dir, 'final_enhanced_model.pth')
        self.save_model(final_path, epoch)
        
        # Final training visualization
        self.save_training_visualizations(epoch)
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Models saved in: {self.save_dir}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1,
            'train_losses': dict(self.train_losses),
            'val_losses': dict(self.val_losses)
        }
    def plot_loss_weights(self, save_path='loss_weights_over_epochs.png'):
        """Plot the evolution of loss weights over epochs"""
        if not hasattr(self, 'loss_weight_log'):
            print("❌ No loss weight logs found.")
            return

        weight_log = np.array(self.loss_weight_log)  # Shape: [epochs, 6]
        epochs = np.arange(1, weight_log.shape[0] + 1)

        plt.figure(figsize=(10, 6))
        for i in range(weight_log.shape[1]):
            plt.plot(epochs, weight_log[:, i], label=f'Weight {i+1}')
    
        plt.title("📊 Evolution of Learnable Loss Weights Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Normalized Weight (Softmax)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Loss weight plot saved to {save_path}")
        plt.close()

def main():
    """Enhanced main function"""
    # Set seed for reproducibility
    set_seed(42)
    
    # Enhanced configuration
    config = {
        'root_train_folder': '/home/pvvkishore/Desktop/TVC_Jan21/New_Code/train/',
        'annotations_folder': '/home/pvvkishore/Desktop/TVC_Jan21/New_Code/annotations/',
        'batch_size': 4,           # Increased batch size
        'max_frames': 30,          # Increased max frames
        'num_epochs': 140,         # More epochs
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'enhanced_checkpoints_1024d',
        'test_split': 0.2,
        'feature_dim': 1024,       # Enhanced to 1024D
        'hidden_dim': 512,         # Enhanced hidden dim
        'lstm_hidden': 256,        # Enhanced LSTM hidden
        'num_heads': 8,            # Multi-head attention
        'dropout': 0.1,
        'num_workers': 2
    }
    
    print("🚀 ENHANCED SIGN LANGUAGE RECOGNITION - 1024D FEATURES")
    print("=" * 70)
    print(f"Enhanced configurations:")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Max frames: {config['max_frames']}")
    print(f"   Feature dim: {config['feature_dim']}D")
    print(f"   Hidden dim: {config['hidden_dim']}")
    print(f"   LSTM hidden: {config['lstm_hidden']}")
    print(f"   Attention heads: {config['num_heads']}")
    print(f"   Device: {config['device']}")
    print()
    
    try:
        # Create enhanced data loaders
        print("Loading enhanced dataset...")
        train_loader, val_loader, full_dataset = get_enhanced_data_loaders(
            root_train_folder=config['root_train_folder'],
            annotations_folder=config['annotations_folder'],
            batch_size=config['batch_size'],
            test_split=config['test_split'],
            max_frames=config['max_frames'],
            num_workers=config['num_workers']
        )
        
        print(f"📊 Dataset loaded:")
        print(f"   Train samples: {len(train_loader.dataset.valid_annotations)}")
        print(f"   Val samples: {len(val_loader.dataset.valid_annotations)}")
        print(f"   Vocabulary size: {len(full_dataset.vocab)}")
        print()
        
        # Create enhanced model
        print("Initializing enhanced model...")
        model = EnhancedFourStageModel(
            vocab_size=len(full_dataset.vocab),
            feature_dim=config['feature_dim'],
            hidden_dim=config['hidden_dim'],
            lstm_hidden=config['lstm_hidden'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        print()
        
        # Create enhanced trainer
        trainer = EnhancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=len(full_dataset.vocab),
            device=config['device'],
            save_dir=config['save_dir']
        )
        
        # Start training
        print("Starting enhanced training...")
        results = trainer.train(num_epochs=config['num_epochs'])
        trainer.plot_loss_weights()

        print("🎉 Enhanced training completed successfully!")
        print(f"📊 Final results: {results}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
