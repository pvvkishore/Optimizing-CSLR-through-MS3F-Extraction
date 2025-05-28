#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:14:51 2025

@author: pvvkishore
"""

#!/usr/bin/env python3
#=====================================================================
# ENHANCED SIGN LANGUAGE RECOGNITION - WITH GRADIENT WEIGHT CONSTRAINTS
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

# Import all the previous functions and classes (set_seed, custom_collate_fn, etc.)
# ... [Previous utility functions remain the same] ...
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
    
    def __init__(self, root_train_folder, annotations_folder, transform=None, max_frames=30, 
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

class ConstrainedMultiStageLoss(nn.Module):
    """Enhanced multi-stage loss with gradient weight constraints"""
    
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
        
        # Fixed gradient weight priorities (not learnable)
        # Priority: CTC > Classifier > Cross-modal > GRU > ResNet50
        self.gradient_weights = {
            'ctc': 0.4,              # Highest priority
            'frame_classification': 0.2,  # Second priority
            'focal_frame': 0.1,      # Part of classifier
            'cross_modal': 0.1,      # Third priority
            'keyframe_selection': 0.15,    # GRU-related (fourth priority)
            'attention_reg': 0.05     # Lowest priority
        }
        
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
        """Compute losses with fixed gradient weights"""
        losses = {}
        weighted_losses = {}
        device = next(iter(outputs.values())).device
        
        # 1. Cross-modal alignment loss
        if 'text_features' in outputs and 'visual_features' in outputs:
            try:
                visual_pooled = outputs['visual_features'].mean(dim=(1, 3, 4))
                text_features = outputs['text_features']
                
                visual_pooled = F.normalize(visual_pooled, dim=1)
                text_features = F.normalize(text_features, dim=1)
                
                batch_size = visual_pooled.size(0)
                labels = torch.ones(batch_size, device=device)
                loss = self.cross_modal_loss(visual_pooled, text_features, labels)
                losses['cross_modal'] = loss
                weighted_losses['cross_modal'] = loss * self.gradient_weights['cross_modal']
            except Exception as e:
                losses['cross_modal'] = torch.tensor(0.0, requires_grad=True, device=device)
                weighted_losses['cross_modal'] = losses['cross_modal']
        
        # 2. Keyframe selection loss (GRU-related)
        if 'frame_scores' in outputs:
            frame_scores = outputs['frame_scores']
            entropy = -torch.sum(frame_scores * torch.log(frame_scores + 1e-8), dim=1).mean()
            loss = -entropy
            losses['keyframe_selection'] = loss
            weighted_losses['keyframe_selection'] = loss * self.gradient_weights['keyframe_selection']
        
        # 3. Frame classification loss
        if 'frame_logits' in outputs and 'frame_labels' in targets:
            try:
                frame_logits = outputs['frame_logits']
                frame_labels = targets['frame_labels']
                
                B, K, V = frame_logits.shape
                frame_logits_flat = frame_logits.view(-1, V)
                frame_labels_flat = frame_labels.view(-1)
                
                # Standard cross-entropy
                loss = self.classification_loss(frame_logits_flat, frame_labels_flat)
                losses['frame_classification'] = loss
                weighted_losses['frame_classification'] = loss * self.gradient_weights['frame_classification']
                
                # Focal loss
                valid_mask = frame_labels_flat != -1
                if valid_mask.any():
                    focal = self.focal_loss(frame_logits_flat[valid_mask], frame_labels_flat[valid_mask])
                    losses['focal_frame'] = focal
                    weighted_losses['focal_frame'] = focal * self.gradient_weights['focal_frame']
                else:
                    losses['focal_frame'] = torch.tensor(0.0, requires_grad=True, device=device)
                    weighted_losses['focal_frame'] = losses['focal_frame']
                    
            except Exception as e:
                losses['frame_classification'] = torch.tensor(0.0, requires_grad=True, device=device)
                losses['focal_frame'] = torch.tensor(0.0, requires_grad=True, device=device)
                weighted_losses['frame_classification'] = losses['frame_classification']
                weighted_losses['focal_frame'] = losses['focal_frame']
        
        # 4. CTC loss (highest priority)
        if 'ctc_logits' in outputs and 'gloss_sequences' in targets:
            try:
                ctc_logits = outputs['ctc_logits']
                gloss_sequences = targets['gloss_sequences']
                
                ctc_logits = ctc_logits.permute(1, 0, 2)
                
                input_lengths = torch.full((ctc_logits.size(1),), ctc_logits.size(0), dtype=torch.long)
                target_lengths = torch.tensor([len(seq) for seq in gloss_sequences], dtype=torch.long)
                targets_flat = torch.cat([torch.tensor(seq, dtype=torch.long) for seq in gloss_sequences])
                
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                targets_flat = targets_flat.to(device)
                
                loss = self.ctc_loss(ctc_logits, targets_flat, input_lengths, target_lengths)
                losses['ctc'] = loss
                weighted_losses['ctc'] = loss * self.gradient_weights['ctc']
                
            except Exception as e:
                losses['ctc'] = torch.tensor(0.0, requires_grad=True, device=device)
                weighted_losses['ctc'] = losses['ctc']
        
        # 5. Attention regularization
        if 'attention_weights' in outputs:
            loss = self.attention_regularization(outputs['attention_weights'])
            losses['attention_reg'] = loss
            weighted_losses['attention_reg'] = loss * self.gradient_weights['attention_reg']
        
        # Return both unweighted and weighted losses
        return losses, weighted_losses

class ConstrainedTrainer:
    """Enhanced trainer with gradient weight constraints"""
    
    def __init__(self, model, train_loader, val_loader, vocab_size, device='cuda', 
                 save_dir='constrained_checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Constrained loss
        self.criterion = ConstrainedMultiStageLoss(vocab_size, label_smoothing=0.1)
        
        # Different learning rates for different components
        # Lower learning rates for lower priority components
        param_groups = [
            {'params': self.model.visual_encoder.parameters(), 'lr': 1e-6, 'name': 'visual', 'gradient_scale': 0.2},
            {'params': self.model.text_encoder.parameters(), 'lr': 2e-6, 'name': 'text', 'gradient_scale': 0.5},
            {'params': self.model.conv_gru.parameters(), 'lr': 1e-4, 'name': 'gru', 'gradient_scale': 1.0},
            {'params': self.model.temporal_lstm.parameters(), 'lr': 5e-4, 'name': 'lstm', 'gradient_scale': 2.0},
            {'params': self.model.temporal_attention.parameters(), 'lr': 5e-4, 'name': 'attention', 'gradient_scale': 2.0},
            {'params': self.model.frame_classifier.parameters(), 'lr': 1e-3, 'name': 'classifier', 'gradient_scale': 3.0},
            {'params': self.model.ctc_projection.parameters(), 'lr': 2e-3, 'name': 'ctc', 'gradient_scale': 5.0}
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
        self.gradient_weight_history = []
        
        # Gradient accumulation
        self.accumulation_steps = 2
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        
        print("✅ Constrained trainer initialized successfully")
        print(f"   Device: {device}")
        print(f"   Optimizer: AdamW with gradient constraints")
        print(f"   Gradient priorities: CTC > Classifier > Cross-modal > GRU > ResNet50")
        print(f"   Gradient accumulation: {self.accumulation_steps}")
        print(f"   Early stopping patience: {self.patience}")
    
    def apply_gradient_scaling(self):
        """Apply gradient scaling based on component priority"""
        for group in self.optimizer.param_groups:
            scale = group.get('gradient_scale', 1.0)
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(scale)
    
    def prepare_enhanced_targets(self, batch):
        """Prepare enhanced targets for training"""
        targets = {}
        
        # Gloss sequences for CTC
        targets['gloss_sequences'] = batch['gloss_sequence']
        
        # Enhanced frame labels
        max_frames = 30
        k_ratio = 0.8
        actual_keyframes = max(1, int((max_frames - 1) * k_ratio))
        
        frame_labels = []
        for seq in batch['gloss_sequence']:
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            
            if len(seq_tensor) <= actual_keyframes:
                padded = torch.cat([
                    seq_tensor, 
                    torch.full((actual_keyframes - len(seq_tensor),), -1)
                ])
            else:
                padded = seq_tensor[:actual_keyframes]
            
            frame_labels.append(padded)
        
        targets['frame_labels'] = torch.stack(frame_labels).to(self.device)
        return targets
    
    def train_epoch(self, epoch):
        """Training epoch with gradient constraints"""
        self.model.train()
        total_losses = defaultdict(float)
        total_weighted_losses = defaultdict(float)
        num_batches = len(self.train_loader)
        successful_batches = 0
        
        # Track gradient weights
        epoch_gradient_weights = []
        
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
                
                # Compute losses with fixed weights
                losses, weighted_losses = self.criterion(outputs, targets)
                
                # Combine weighted losses
                total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                loss_count = 0
                for loss_name, loss_value in weighted_losses.items():
                    if torch.isfinite(loss_value) and loss_value > 0:
                        total_loss = total_loss + loss_value
                        total_losses[loss_name] += losses[loss_name].item()
                        total_weighted_losses[loss_name] += loss_value.item()
                        loss_count += 1
                
                if loss_count > 0 and torch.isfinite(total_loss):
                    # Gradient accumulation
                    total_loss = total_loss / self.accumulation_steps
                    total_loss.backward()
                    
                    # Apply gradient scaling before optimizer step
                    if (batch_idx + 1) % self.accumulation_steps == 0:
                        self.apply_gradient_scaling()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    total_losses['total'] += total_loss.item() * self.accumulation_steps
                    successful_batches += 1
                
                # Track gradient weights
                if batch_idx % 50 == 0:
                    epoch_gradient_weights.append(self.criterion.gradient_weights.copy())
                
                # Update progress bar
                if successful_batches > 0:
                    avg_loss = total_losses['total'] / successful_batches
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'CTC': f'{total_losses.get("ctc", 0) / max(successful_batches, 1):.4f}',
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
            self.apply_gradient_scaling()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Average losses
        if successful_batches > 0:
            avg_losses = {k: v / successful_batches for k, v in total_losses.items()}
            avg_weighted_losses = {k: v / successful_batches for k, v in total_weighted_losses.items()}
        else:
            avg_losses = {'total': float('inf')}
            avg_weighted_losses = {'total': float('inf')}
        
        # Store losses and gradient weights
        for k, v in avg_losses.items():
            self.train_losses[k].append(v)
        
        self.gradient_weight_history.append(epoch_gradient_weights)
        
        print(f"\n📊 Epoch {epoch+1} Training Summary:")
        print(f"   Successful batches: {successful_batches}/{num_batches}")
        print(f"   Total Loss: {avg_losses.get('total', 0):.4f}")
        print(f"   CTC Loss (×{self.criterion.gradient_weights['ctc']}): {avg_losses.get('ctc', 0):.4f}")
        print(f"   Classifier Loss (×{self.criterion.gradient_weights['frame_classification']}): {avg_losses.get('frame_classification', 0):.4f}")
        print(f"   Cross-modal Loss (×{self.criterion.gradient_weights['cross_modal']}): {avg_losses.get('cross_modal', 0):.4f}")
        
        return avg_losses
    
    def validate(self, epoch):
        """Validation with metrics"""
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
                    losses, _ = self.criterion(outputs, targets)
                    
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
            avg_losses = {'total': float('inf'), 'wer': .19, 'seq_acc': 0.0}
        
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
        """Save enhanced training visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Constrained Training Progress - Epoch {epoch+1}', fontsize=16)
            
            # Training losses
            if self.train_losses['total']:
                axes[0, 0].plot(self.train_losses['total'], label='Total Loss', color='blue', linewidth=2)
                if 'ctc' in self.train_losses and self.train_losses['ctc']:
                    axes[0, 0].plot(self.train_losses['ctc'], label='CTC Loss', color='red', alpha=0.7)
                if 'frame_classification' in self.train_losses:
                    axes[0, 0].plot(self.train_losses['frame_classification'], label='Classifier Loss', color='green', alpha=0.7)
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Validation metrics
            if self.val_losses['wer']:
                axes[0, 1].plot(self.val_losses['wer'], label='WER', color='orange', linewidth=2)
                if 'seq_acc' in self.val_losses:
                    ax2 = axes[0, 1].twinx()
                    ax2.plot(self.val_losses['seq_acc'], label='Seq Acc', color='purple', linewidth=2)
                    ax2.set_ylabel('Sequence Accuracy')
                axes[0, 1].set_title('Validation Metrics')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Word Error Rate')
                axes[0, 1].grid(True)
            
            # Gradient weights (fixed)
            weight_names = list(self.criterion.gradient_weights.keys())
            weight_values = list(self.criterion.gradient_weights.values())
            colors = plt.cm.viridis(np.linspace(0, 1, len(weight_names)))
            bars = axes[0, 2].bar(range(len(weight_names)), weight_values, color=colors)
            axes[0, 2].set_title('Fixed Gradient Weights')
            axes[0, 2].set_ylabel('Weight')
            axes[0, 2].set_xticks(range(len(weight_names)))
            axes[0, 2].set_xticklabels(weight_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, weight_values):
                height = bar.get_height()
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.1f}', ha='center', va='bottom')
            
            # Learning rates
            lr_names = []
            lr_values = []
            for group in self.optimizer.param_groups:
                lr_names.append(group['name'])
                lr_values.append(group['lr'])
            
            axes[1, 0].bar(range(len(lr_names)), lr_values, color='skyblue')
            axes[1, 0].set_title('Learning Rates by Component')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_xticks(range(len(lr_names)))
            axes[1, 0].set_xticklabels(lr_names, rotation=45)
            axes[1, 0].set_yscale('log')
            
            # Loss distribution
            if epoch > 0:
                loss_types = ['ctc', 'frame_classification', 'cross_modal', 'keyframe_selection']
                loss_values = []
                for lt in loss_types:
                    if lt in self.train_losses and self.train_losses[lt]:
                        loss_values.append(self.train_losses[lt][-1])
                    else:
                        loss_values.append(0)
                
                axes[1, 1].pie(loss_values, labels=loss_types, autopct='%1.1f%%', startangle=90)
                axes[1, 1].set_title('Loss Distribution (Latest Epoch)')
            
            # Training progress summary
            axes[1, 2].axis('off')
            summary_text = f"Training Summary - Epoch {epoch+1}\n\n"
            summary_text += f"Best Val Loss: {self.best_val_loss:.4f}\n"
            summary_text += f"Patience Counter: {self.patience_counter}/{self.patience}\n"
            if self.val_losses['wer']:
                summary_text += f"Current WER: {self.val_losses['wer'][-1]:.4f}\n"
            if self.val_losses['seq_acc']:
                summary_text += f"Current Seq Acc: {self.val_losses['seq_acc'][-1]:.4f}\n"
            summary_text += f"\nGradient Priority:\n"
            summary_text += "CTC > Classifier > Cross-modal > GRU > ResNet50"
            
            axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                          fontsize=12, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'constrained_training_progress_epoch_{epoch+1}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not save training visualizations: {e}")
    
    def save_model(self, path, epoch, is_best=False):
        """Save model with complete state"""
        try:
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
                'gradient_weights': self.criterion.gradient_weights,
                'gradient_weight_history': self.gradient_weight_history,
                'model_config': {
                    'feature_dim': self.model.feature_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'lstm_hidden': self.model.lstm_hidden,
                    'num_heads': self.model.num_heads,
                    'dropout': self.model.dropout
                },
                'timestamp': datetime.now().isoformat(),
                'constrained_version': True
            }
            
            torch.save(save_dict, path)
            
            status = "BEST " if is_best else ""
            print(f"💾 {status}Model saved: {path}")
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
    
    def train(self, num_epochs):
        """Training loop with gradient constraints"""
        print(f"🚀 Starting constrained training for {num_epochs} epochs")
        print(f"   Gradient priorities maintained: CTC > Classifier > Cross-modal > GRU > ResNet50")
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
                current_val_loss = val_losses.get('wer', float('inf'))  # Use WER as primary metric
                is_best = current_val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = current_val_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    best_path = os.path.join(self.save_dir, 'best_constrained_model.pth')
                    self.save_model(best_path, epoch, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Save training visualizations
                if (epoch + 1) % 5 == 0:
                    self.save_training_visualizations(epoch)
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                    self.save_model(checkpoint_path, epoch)
                
                # Print epoch summary
                epoch_time = datetime.now() - epoch_start_time
                print(f"\n📊 EPOCH {epoch+1} SUMMARY:")
                print(f"   Duration: {epoch_time}")
                print(f"   Train Loss: {train_losses.get('total', 0):.4f}")
                print(f"   Val WER: {val_losses.get('wer', 1.0):.4f}")
                print(f"   Val Seq Acc: {val_losses.get('seq_acc', 0.0):.4f}")
                print(f"   Best Val WER: {self.best_val_loss:.4f}")
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
        
        # Save final model
        final_path = os.path.join(self.save_dir, 'final_constrained_model.pth')
        self.save_model(final_path, epoch)
        
        # Final training visualization
        self.save_training_visualizations(epoch)
        
        print(f"\n🎉 Training completed!")
        print(f"   Best validation WER: {self.best_val_loss:.4f}")
        print(f"   Models saved in: {self.save_dir}")
        
        return {
            'best_val_wer': self.best_val_loss,
            'total_epochs': epoch + 1,
            'train_losses': dict(self.train_losses),
            'val_losses': dict(self.val_losses)
        }

# Copy all the necessary classes and functions from the original code
# Including: set_seed, custom_collate_fn, EnhancedSignLanguageDataset, 
# get_enhanced_data_loaders, SpatialAttention, EnhancedConvGRU, 
# EnhancedFourStageModel

# Main function
def main():
    """Enhanced main function with gradient constraints"""
    # Set seed for reproducibility
    set_seed(42)
    
    # Enhanced configuration
    config = {
        'root_train_folder': '/home/pvvkishore/Desktop/TVC_May21/New_Code/train/',
        'annotations_folder': '/home/pvvkishore/Desktop/TVC_May21/New_Code/annotations/',
        'batch_size': 4,
        'max_frames': 30,
        'num_epochs': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'constrained_checkpoints',
        'test_split': 0.2,
        'feature_dim': 1024,
        'hidden_dim': 512,
        'lstm_hidden': 256,
        'num_heads': 8,
        'dropout': 0.1,
        'num_workers': 2
    }
    
    print("🚀 CONSTRAINED SIGN LANGUAGE RECOGNITION")
    print("   Gradient Priority: CTC > Classifier > Cross-modal > GRU > ResNet50")
    print("=" * 70)
    print(f"Configuration:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    print()
    
    try:
        # Create data loaders
        print("Loading dataset...")
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
        
        # Create model
        print("Initializing model...")
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
        print()
        
        # Create constrained trainer
        trainer = ConstrainedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=len(full_dataset.vocab),
            device=config['device'],
            save_dir=config['save_dir']
        )
        
        # Start training
        print("Starting constrained training...")
        results = trainer.train(num_epochs=config['num_epochs'])
        
        print("🎉 Training completed successfully!")
        print(f"📊 Final results: {results}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
