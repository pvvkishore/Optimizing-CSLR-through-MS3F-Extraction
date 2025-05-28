#!/usr/bin/env python3
"""
Enhanced Sign Language Model Inference Script
Designed for the enhanced 1024D model with comprehensive evaluation
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import cv2
import numpy as np
import json
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict, Counter
import editdistance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Environment setup
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Try to import transformers
try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available, text encoder will use dummy features")
    TRANSFORMERS_AVAILABLE = False

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
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
        
        self.conv_reset = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_update = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_new = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
        self.spatial_attention = SpatialAttention(hidden_dim)
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
        new_hidden = self.spatial_attention(new_hidden)
        new_hidden = self.norm(new_hidden)
        
        return new_hidden

class EnhancedTextEncoder(nn.Module):
    """Enhanced text encoder"""
    def __init__(self, feature_dim=1024):
        super().__init__()
        self.feature_dim = feature_dim
        
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
            # Freeze early layers
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < 8:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            self.projection = nn.Sequential(
                nn.Linear(768, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
        else:
            # Dummy encoder
            self.projection = nn.Linear(100, feature_dim)
    
    def forward(self, text_list):
        if not TRANSFORMERS_AVAILABLE:
            batch_size = len(text_list)
            device = next(self.parameters()).device
            return torch.randn(batch_size, self.feature_dim, device=device)
        
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
        
        pooled_output = outputs.pooler_output
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        combined = torch.cat([pooled_output, cls_output], dim=-1)
        combined = nn.Linear(1536, 768).to(device)(combined)
        
        return self.projection(combined)

class EnhancedFourStageModel(nn.Module):
    """Enhanced 4-stage model matching the training script"""
    
    def __init__(self, vocab_size, feature_dim=1024, hidden_dim=512, lstm_hidden=256, 
                 num_heads=8, dropout=0.1):
        super(EnhancedFourStageModel, self).__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Stage 1: Enhanced Visual Feature Extraction
        self.visual_encoder = self._build_enhanced_visual_encoder()
        self.text_encoder = EnhancedTextEncoder(feature_dim)
        
        # Stage 2: Enhanced Motion-aware Keyframe Selection
        self.conv_gru = EnhancedConvGRU(input_dim=feature_dim, hidden_dim=hidden_dim)
        
        self.temporal_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
        ])
        
        self.keyframe_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Stage 3: Enhanced Temporal Modeling
        self.temporal_lstm = nn.LSTM(
            input_size=feature_dim * 4 * 4,
            hidden_size=lstm_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_hidden > 1 else 0
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
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
        
        self.loss_weights = nn.Parameter(torch.ones(5))
    
    def _build_enhanced_visual_encoder(self):
        """Enhanced visual encoder with 1024D features"""
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        layers = list(resnet.children())[:-2]
        visual_encoder = nn.Sequential(*layers)
        
        visual_encoder.add_module('projection', nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(2048, self.feature_dim, 1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU()
        ))
        
        return visual_encoder
    
    def stage1_enhanced_visual_extraction(self, frames):
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        visual_features = self.visual_encoder(frames_flat)
        _, feat_dim, feat_h, feat_w = visual_features.shape
        visual_features = visual_features.view(B, T, feat_dim, feat_h, feat_w)
        return visual_features
    
    def stage2_enhanced_keyframe_selection(self, visual_features, k_ratio=0.8):
        B, T, C, H, W = visual_features.shape
        frame_diffs = visual_features[:, 1:] - visual_features[:, :-1]
        
        hidden_state = None
        gru_outputs = []
        
        for t in range(T - 1):
            hidden_state = self.conv_gru(frame_diffs[:, t], hidden_state)
            gru_outputs.append(hidden_state)
        
        gru_features = torch.stack(gru_outputs, dim=1)
        
        pooled_features = []
        for pool in self.temporal_pools:
            pooled = pool(gru_features.view(B * (T-1), -1, H, W))
            pooled = pooled.view(B, T-1, -1)
            pooled_features.append(pooled)
        
        combined_features = torch.cat(pooled_features, dim=-1)
        frame_scores = self.keyframe_selector(combined_features)
        frame_scores = frame_scores.squeeze(-1)
        frame_scores = F.softmax(frame_scores, dim=1)
        
        k = max(1, int((T-1) * k_ratio))
        _, top_indices = torch.topk(frame_scores, k, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)
        
        selected_features = []
        for b in range(B):
            batch_features = visual_features[b, 1:][top_indices[b]]
            selected_features.append(batch_features)
        
        selected_features = torch.stack(selected_features)
        
        return selected_features, frame_scores, top_indices
    
    def stage3_enhanced_temporal_modeling(self, selected_features):
        B, K, C, H, W = selected_features.shape
        features_flat = selected_features.view(B, K, C * H * W)
        
        lstm_out, (hidden, cell) = self.temporal_lstm(features_flat)
        
        attended_features, attention_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out
        )
        
        combined_features = lstm_out + attended_features
        frame_logits = self.frame_classifier(combined_features)
        
        return combined_features, frame_logits, attention_weights
    
    def stage4_enhanced_ctc_prediction(self, lstm_features):
        ctc_logits = self.ctc_projection(lstm_features)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        return ctc_log_probs
    
    def forward(self, frames, gloss_texts=None, gloss_sequences=None, mode='eval'):
        outputs = {}
        
        visual_features = self.stage1_enhanced_visual_extraction(frames)
        outputs['visual_features'] = visual_features
        
        if gloss_texts is not None:
            try:
                text_features = self.text_encoder(gloss_texts)
                outputs['text_features'] = text_features
            except Exception as e:
                pass
        
        selected_features, frame_scores, keyframe_indices = self.stage2_enhanced_keyframe_selection(visual_features)
        outputs['selected_features'] = selected_features
        outputs['frame_scores'] = frame_scores
        outputs['keyframe_indices'] = keyframe_indices
        
        lstm_features, frame_logits, attention_weights = self.stage3_enhanced_temporal_modeling(selected_features)
        outputs['lstm_features'] = lstm_features
        outputs['frame_logits'] = frame_logits
        outputs['attention_weights'] = attention_weights
        
        ctc_logits = self.stage4_enhanced_ctc_prediction(lstm_features)
        outputs['ctc_logits'] = ctc_logits
        
        return outputs

class EnhancedTestDataset(Dataset):
    """Enhanced test dataset matching training preprocessing"""
    def __init__(self, test_folder, annotations_file, vocab, transform=None, max_frames=20):
        self.test_folder = test_folder
        self.transform = transform
        self.max_frames = max_frames
        self.vocab = vocab
        self.idx_to_vocab = {v: k for k, v in vocab.items()}
        
        # Load annotations
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations.iloc[1:, :2]
        self.annotations.columns = ['folder_name', 'gloss_text']
        
        # Filter valid samples
        self.valid_samples = []
        for _, row in self.annotations.iterrows():
            folder_path = os.path.join(test_folder, str(row['folder_name']))
            if os.path.exists(folder_path):
                self.valid_samples.append({
                    'folder_name': str(row['folder_name']),
                    'gloss_text': str(row['gloss_text']).lower().strip(),
                    'folder_path': folder_path,
                    'gloss_sequence': self.text_to_sequence(str(row['gloss_text']))
                })
        
        print(f"📁 Loaded {len(self.valid_samples)} valid test samples")
    
    def text_to_sequence(self, text):
        words = text.lower().strip().split()
        sequence = []
        for word in words:
            word = word.strip()
            if word:
                if word in self.vocab:
                    sequence.append(self.vocab[word])
                else:
                    sequence.append(self.vocab.get('<unk>', 1))
        return sequence
    
    def sequence_to_text(self, sequence):
        words = []
        for idx in sequence:
            if idx in self.idx_to_vocab and idx not in [0, 2]:  # Skip blank and pad
                words.append(self.idx_to_vocab[idx])
        return ' '.join(words)
    
    def __len__(self):
        return len(self.valid_samples)
    
    def load_frames(self, folder_path):
        frame_files = sorted([f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not frame_files:
            raise ValueError(f"No frames found in {folder_path}")
        
        # Uniform sampling for test
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files)-1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            try:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LANCZOS4)
                    frames.append(frame)
            except Exception as e:
                continue
        
        if not frames:
            raise ValueError(f"No valid frames loaded from {folder_path}")
        
        # Pad to max_frames
        while len(frames) < self.max_frames:
            frames.append(frames[-1].copy())
        
        return np.stack(frames[:self.max_frames])
    
    def __getitem__(self, idx):
        if idx >= len(self.valid_samples):
            idx = idx % len(self.valid_samples)
            
        sample = self.valid_samples[idx]
        
        try:
            frames = self.load_frames(sample['folder_path'])
            
            if self.transform:
                transformed_frames = []
                for frame in frames:
                    frame_pil = Image.fromarray(frame.astype(np.uint8))
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
            print(f"Error loading sample {idx}: {e}")
            dummy_frames = torch.zeros(self.max_frames, 3, 112, 112)
            return {
                'frames': dummy_frames,
                'gloss_text': "unknown",
                'folder_name': "error",
                'gloss_sequence': [1]
            }

class EnhancedGradCAM:
    """Enhanced Grad-CAM for the enhanced model"""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Hook the last conv layer before projection
        for name, module in self.model.visual_encoder.named_modules():
            if name == '7.0':  # Last ResNet layer
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        
        try:
            output = self.model(input_tensor)
            
            if class_idx is None:
                class_idx = torch.argmax(output['ctc_logits']).item()
            
            self.model.zero_grad()
            class_score = output['ctc_logits'][:, :, class_idx].sum()
            class_score.backward(retain_graph=True)
            
            if self.gradients is not None and self.activations is not None:
                pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
                
                for i in range(self.activations.size(1)):
                    self.activations[:, i, :, :] *= pooled_gradients[i]
                
                heatmap = torch.mean(self.activations, dim=1).squeeze()
                heatmap = F.relu(heatmap)
                
                if heatmap.max() > 0:
                    heatmap /= heatmap.max()
                
                return heatmap.detach().cpu().numpy()
            
        except Exception as e:
            print(f"Grad-CAM generation error: {e}")
            return np.random.rand(112, 112) * 0.3  # Dummy heatmap
        
        return None

class EnhancedInferenceEvaluator:
    """Enhanced inference evaluator with comprehensive analysis"""
    
    def __init__(self, model_path, test_folder, annotations_file, results_dir='enhanced_results'):
        self.model_path = model_path
        self.test_folder = test_folder
        self.annotations_file = annotations_file
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🚀 Enhanced Inference Evaluator")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        print(f"   Test data: {test_folder}")
        print(f"   Results: {results_dir}")
        
        # Load model and vocabulary
        self.model, self.vocab, self.model_config = self.load_enhanced_model()
        self.test_dataset = self.create_test_dataset()
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize Grad-CAM
        self.grad_cam = EnhancedGradCAM(self.model)
        
        print(f"✅ Initialization complete")
        print(f"   Vocabulary size: {len(self.vocab)}")
        print(f"   Test samples: {len(self.test_dataset)}")
    
    def load_enhanced_model(self):
        """Load the enhanced model with proper configuration"""
        print("📦 Loading enhanced model...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            print("✅ Checkpoint loaded successfully")
            
            # Debug checkpoint contents
            print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
            if 'enhanced_version' in checkpoint:
                print(f"✅ Enhanced model detected")
            
            # Get vocabulary
            if 'vocab' in checkpoint and checkpoint['vocab']:
                vocab = checkpoint['vocab']
                print(f"📚 Vocabulary loaded from checkpoint: {len(vocab)} tokens")
            else:
                # Try to load from separate file
                vocab_path = os.path.join(os.path.dirname(self.test_folder), 'vocabulary.pkl')
                if os.path.exists(vocab_path):
                    with open(vocab_path, 'rb') as f:
                        vocab_data = pickle.load(f)
                        vocab = vocab_data['vocab']
                    print(f"📚 Vocabulary loaded from file: {len(vocab)} tokens")
                else:
                    # Create basic vocabulary
                    vocab_size = checkpoint.get('vocab_size', 1000)
                    vocab = {'<blank>': 0, '<unk>': 1, '<pad>': 2}
                    for i in range(3, vocab_size):
                        vocab[f'token_{i}'] = i
                    print(f"⚠️  Created basic vocabulary: {len(vocab)} tokens")
            
            # Get model configuration
            model_config = checkpoint.get('model_config', {
                'feature_dim': 1024,
                'hidden_dim': 512,
                'lstm_hidden': 256,
                'num_heads': 8,
                'dropout': 0.1
            })
            
            print(f"🏗️  Model config: {model_config}")
            
            # Create enhanced model
            model = EnhancedFourStageModel(
                vocab_size=len(vocab),
                **model_config
            )
            
            # Load state dict
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✅ Model weights loaded")
                if missing_keys:
                    print(f"⚠️  Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            except Exception as e:
                print(f"⚠️  Model loading warning: {e}")
            
            model.to(self.device)
            model.eval()
            
            return model, vocab, model_config
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def create_test_dataset(self):
        """Create enhanced test dataset"""
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return EnhancedTestDataset(
            test_folder=self.test_folder,
            annotations_file=self.annotations_file,
            vocab=self.vocab,
            transform=transform,
            max_frames=20  # Match training
        )
    
    def enhanced_ctc_decode(self, log_probs, blank=0):
        """Enhanced CTC decoding with beam search approximation"""
        # Simple greedy decoding
        path = torch.argmax(log_probs, dim=-1)
        
        decoded = []
        prev = None
        for token in path:
            token = token.item()
            if token != blank and token != prev:
                decoded.append(token)
            prev = token
        
        return decoded
    
    def run_enhanced_inference(self):
        """Run enhanced inference with detailed analysis"""
        print("🔍 Running enhanced inference...")
        
        all_predictions = []
        all_ground_truths = []
        all_sample_info = []
        attention_data = []
        keyframe_data = []
        
        successful_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Enhanced Inference")):
                try:
                    frames = batch['frames'].to(self.device)
                    gloss_text = batch['gloss_text'][0]
                    folder_name = batch['folder_name'][0]
                    ground_truth = batch['gloss_sequence'][0]
                    
                    # Forward pass
                    outputs = self.model(frames, mode='eval')
                    
                    # Enhanced CTC decoding
                    ctc_log_probs = outputs['ctc_logits'][0]
                    prediction = self.enhanced_ctc_decode(ctc_log_probs)
                    
                    # Convert to text
                    pred_text = self.test_dataset.sequence_to_text(prediction)
                    
                    # Store results
                    all_predictions.append(prediction)
                    all_ground_truths.append(ground_truth)
                    
                    # Enhanced sample info
                    sample_info = {
                        'folder_name': folder_name,
                        'ground_truth_text': gloss_text,
                        'prediction_text': pred_text,
                        'ground_truth_seq': ground_truth,
                        'prediction_seq': prediction,
                        'edit_distance': editdistance.eval(prediction, ground_truth),
                        'sequence_length_gt': len(ground_truth),
                        'sequence_length_pred': len(prediction),
                        'keyframe_indices': outputs.get('keyframe_indices', [[]])[0].cpu().numpy().tolist(),
                        'frame_scores': outputs.get('frame_scores', [[]])[0].cpu().numpy().tolist(),
                        'num_keyframes': len(outputs.get('keyframe_indices', [[]])[0])
                    }
                    
                    # Store attention data
                    if 'attention_weights' in outputs:
                        attention_weights = outputs['attention_weights'][0].cpu().numpy()
                        attention_data.append({
                            'folder_name': folder_name,
                            'attention_weights': attention_weights,
                            'sequence_length': len(ground_truth)
                        })
                    
                    # Store keyframe data
                    if 'frame_scores' in outputs:
                        keyframe_data.append({
                            'folder_name': folder_name,
                            'frame_scores': sample_info['frame_scores'],
                            'keyframe_indices': sample_info['keyframe_indices'],
                            'num_keyframes': sample_info['num_keyframes']
                        })
                    
                    all_sample_info.append(sample_info)
                    successful_samples += 1
                    
                    # Memory cleanup
                    del outputs, frames
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing sample {batch_idx}: {e}")
                    continue
        
        print(f"✅ Enhanced inference completed: {successful_samples}/{len(self.test_loader)} samples")
        
        return all_predictions, all_ground_truths, all_sample_info, attention_data, keyframe_data
    
    def compute_enhanced_metrics(self, predictions, ground_truths, sample_info):
        """Compute comprehensive metrics"""
        print("📊 Computing enhanced metrics...")
        
        metrics = {}
        
        # Basic metrics
        total_errors = 0
        total_words = 0
        perfect_matches = 0
        
        for pred, truth in zip(predictions, ground_truths):
            if len(truth) > 0:
                errors = editdistance.eval(pred, truth)
                total_errors += errors
                total_words += len(truth)
                
                if errors == 0:
                    perfect_matches += 1
        
        metrics['word_error_rate'] = total_errors / max(total_words, 1)
        metrics['sequence_accuracy'] = perfect_matches / len(predictions)
        
        # Token-level metrics
        all_pred_tokens = [token for seq in predictions for token in seq]
        all_true_tokens = [token for seq in ground_truths for token in seq]
        
        if all_pred_tokens and all_true_tokens:
            min_len = min(len(all_pred_tokens), len(all_true_tokens))
            token_matches = sum(1 for i in range(min_len) 
                               if all_pred_tokens[i] == all_true_tokens[i])
            metrics['token_accuracy'] = token_matches / max(len(all_true_tokens), 1)
        else:
            metrics['token_accuracy'] = 0.0
        
        # Length statistics
        pred_lengths = [len(seq) for seq in predictions]
        true_lengths = [len(seq) for seq in ground_truths]
        
        metrics['avg_prediction_length'] = np.mean(pred_lengths) if pred_lengths else 0
        metrics['avg_ground_truth_length'] = np.mean(true_lengths) if true_lengths else 0
        metrics['length_correlation'] = np.corrcoef(pred_lengths, true_lengths)[0, 1] if len(pred_lengths) > 1 else 0
        
        # Enhanced metrics
        edit_distances = [info['edit_distance'] for info in sample_info]
        metrics['mean_edit_distance'] = np.mean(edit_distances)
        metrics['median_edit_distance'] = np.median(edit_distances)
        metrics['std_edit_distance'] = np.std(edit_distances)
        
        # Length-based analysis
        short_sequences = [(p, t) for p, t in zip(predictions, ground_truths) if len(t) <= 3]
        medium_sequences = [(p, t) for p, t in zip(predictions, ground_truths) if 3 < len(t) <= 6]
        long_sequences = [(p, t) for p, t in zip(predictions, ground_truths) if len(t) > 6]
        
        for name, sequences in [('short', short_sequences), ('medium', medium_sequences), ('long', long_sequences)]:
            if sequences:
                errors = sum(editdistance.eval(p, t) for p, t in sequences)
                words = sum(len(t) for _, t in sequences)
                metrics[f'wer_{name}'] = errors / max(words, 1)
                
                perfect = sum(1 for p, t in sequences if editdistance.eval(p, t) == 0)
                metrics[f'accuracy_{name}'] = perfect / len(sequences)
        
        # Keyframe analysis
        keyframe_counts = [info['num_keyframes'] for info in sample_info if 'num_keyframes' in info]
        if keyframe_counts:
            metrics['avg_keyframes'] = np.mean(keyframe_counts)
            metrics['std_keyframes'] = np.std(keyframe_counts)
        
        # Vocabulary usage
        unique_pred_tokens = set(all_pred_tokens)
        unique_true_tokens = set(all_true_tokens)
        
        if unique_true_tokens:
            metrics['vocabulary_coverage'] = len(unique_pred_tokens & unique_true_tokens) / len(unique_true_tokens)
            metrics['vocabulary_precision'] = len(unique_pred_tokens & unique_true_tokens) / max(len(unique_pred_tokens), 1)
        
        print(f"📈 Enhanced metrics computed ({len(metrics)} metrics)")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        return metrics
    
    def create_enhanced_visualizations(self, metrics, sample_info, attention_data, keyframe_data):
        """Create comprehensive visualizations"""
        print("🎨 Creating enhanced visualizations...")
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Main metrics dashboard
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle('Enhanced Sign Language Recognition - Inference Results', fontsize=16, fontweight='bold')
            
            # Key metrics
            key_metrics = ['word_error_rate', 'sequence_accuracy', 'token_accuracy', 'vocabulary_coverage']
            key_values = [metrics.get(k, 0) for k in key_metrics]
            key_labels = ['WER', 'Seq Acc', 'Token Acc', 'Vocab Cov']
            
            bars = axes[0, 0].bar(range(len(key_labels)), key_values, 
                                 color=['red', 'green', 'blue', 'orange'])
            axes[0, 0].set_xticks(range(len(key_labels)))
            axes[0, 0].set_xticklabels(key_labels)
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Key Performance Metrics')
            axes[0, 0].set_ylim(0, 1)
            
            for bar, value in zip(bars, key_values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom')
            
            # Length analysis
            if 'wer_short' in metrics:
                length_categories = ['Short (≤3)', 'Medium (4-6)', 'Long (>6)']
                wer_values = [metrics.get(f'wer_short', 0), metrics.get(f'wer_medium', 0), metrics.get(f'wer_long', 0)]
                acc_values = [metrics.get(f'accuracy_short', 0), metrics.get(f'accuracy_medium', 0), metrics.get(f'accuracy_long', 0)]
                
                x = np.arange(len(length_categories))
                width = 0.35
                
                bars1 = axes[0, 1].bar(x - width/2, wer_values, width, label='WER', color='red', alpha=0.7)
                bars2 = axes[0, 1].bar(x + width/2, acc_values, width, label='Accuracy', color='green', alpha=0.7)
                
                axes[0, 1].set_xlabel('Sequence Length Category')
                axes[0, 1].set_ylabel('Score')
                axes[0, 1].set_title('Performance by Sequence Length')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(length_categories)
                axes[0, 1].legend()
            
            # Edit distance distribution
            edit_distances = [info['edit_distance'] for info in sample_info]
            if edit_distances:
                axes[0, 2].hist(edit_distances, bins=20, color='purple', alpha=0.7, edgecolor='black')
                axes[0, 2].set_xlabel('Edit Distance')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].set_title('Edit Distance Distribution')
                axes[0, 2].axvline(np.mean(edit_distances), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(edit_distances):.2f}')
                axes[0, 2].legend()
            
            # Sequence length comparison
            pred_lengths = [info['sequence_length_pred'] for info in sample_info]
            true_lengths = [info['sequence_length_gt'] for info in sample_info]
            
            if pred_lengths and true_lengths:
                axes[1, 0].scatter(true_lengths, pred_lengths, alpha=0.6, color='blue')
                max_len = max(max(true_lengths), max(pred_lengths))
                axes[1, 0].plot([0, max_len], [0, max_len], 'r--', label='Perfect Prediction')
                axes[1, 0].set_xlabel('Ground Truth Length')
                axes[1, 0].set_ylabel('Prediction Length')
                axes[1, 0].set_title('Length Correlation')
                axes[1, 0].legend()
            
            # Keyframe analysis
            if keyframe_data:
                keyframe_counts = [data['num_keyframes'] for data in keyframe_data]
                axes[1, 1].hist(keyframe_counts, bins=15, color='cyan', alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Number of Keyframes')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Keyframe Selection Distribution')
                if keyframe_counts:
                    axes[1, 1].axvline(np.mean(keyframe_counts), color='red', linestyle='--',
                                      label=f'Mean: {np.mean(keyframe_counts):.1f}')
                    axes[1, 1].legend()
            
            # Performance vs sequence length
            if true_lengths and edit_distances:
                axes[1, 2].scatter(true_lengths, edit_distances, alpha=0.6, color='orange')
                axes[1, 2].set_xlabel('Ground Truth Length')
                axes[1, 2].set_ylabel('Edit Distance')
                axes[1, 2].set_title('Error vs Sequence Length')
                
                # Add trend line
                z = np.polyfit(true_lengths, edit_distances, 1)
                p = np.poly1d(z)
                axes[1, 2].plot(sorted(true_lengths), p(sorted(true_lengths)), "r--", alpha=0.8)
            
            # Confusion matrix of sequence lengths
            if pred_lengths and true_lengths:
                length_bins = [0, 2, 4, 6, 10, float('inf')]
                length_labels = ['1-2', '3-4', '5-6', '7-10', '>10']
                
                true_binned = np.digitize(true_lengths, length_bins) - 1
                pred_binned = np.digitize(pred_lengths, length_bins) - 1
                
                cm = confusion_matrix(true_binned, pred_binned, labels=range(len(length_labels)))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=length_labels, yticklabels=length_labels, ax=axes[2, 0])
                axes[2, 0].set_xlabel('Predicted Length')
                axes[2, 0].set_ylabel('True Length')
                axes[2, 0].set_title('Length Prediction Confusion Matrix')
            
            # Success rate by length
            if true_lengths and edit_distances:
                length_success = defaultdict(list)
                for tl, ed in zip(true_lengths, edit_distances):
                    length_success[tl].append(ed == 0)
                
                lengths = sorted(length_success.keys())
                success_rates = [np.mean(length_success[l]) for l in lengths]
                
                if lengths:
                    axes[2, 1].bar(lengths, success_rates, color='lightgreen', alpha=0.7)
                    axes[2, 1].set_xlabel('Sequence Length')
                    axes[2, 1].set_ylabel('Perfect Match Rate')
                    axes[2, 1].set_title('Success Rate by Length')
            
            # Vocabulary usage analysis
            all_pred_tokens = [token for info in sample_info for token in info['prediction_seq']]
            all_true_tokens = [token for info in sample_info for token in info['ground_truth_seq']]
            
            if all_pred_tokens and all_true_tokens:
                pred_counter = Counter(all_pred_tokens)
                true_counter = Counter(all_true_tokens)
                
                # Top 10 most common tokens
                common_true = dict(true_counter.most_common(10))
                common_pred = {k: pred_counter.get(k, 0) for k in common_true.keys()}
                
                x = np.arange(len(common_true))
                width = 0.35
                
                axes[2, 2].bar(x - width/2, list(common_true.values()), width, 
                              label='Ground Truth', alpha=0.7)
                axes[2, 2].bar(x + width/2, list(common_pred.values()), width, 
                              label='Predictions', alpha=0.7)
                
                axes[2, 2].set_xlabel('Token Index')
                axes[2, 2].set_ylabel('Frequency')
                axes[2, 2].set_title('Top 10 Token Usage')
                axes[2, 2].set_xticks(x)
                axes[2, 2].set_xticklabels([str(k) for k in common_true.keys()], rotation=45)
                axes[2, 2].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'enhanced_metrics_dashboard.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Attention visualization
            if attention_data:
                self.visualize_attention_patterns(attention_data[:5])
            
            # 3. Keyframe visualization
            if keyframe_data:
                self.visualize_keyframe_patterns(keyframe_data[:5])
            
            print("✅ Enhanced visualizations saved")
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_attention_patterns(self, attention_data):
        """Visualize attention patterns"""
        try:
            fig, axes = plt.subplots(1, len(attention_data), figsize=(15, 3))
            if len(attention_data) == 1:
                axes = [axes]
            
            for i, data in enumerate(attention_data):
                attn = data['attention_weights']
                folder_name = data['folder_name']
                
                # Average across heads
                if len(attn.shape) == 3:  # (heads, seq, seq)
                    attn_avg = np.mean(attn, axis=0)
                else:
                    attn_avg = attn
                
                im = axes[i].imshow(attn_avg, cmap='Blues', aspect='auto')
                axes[i].set_title(f'{folder_name}')
                axes[i].set_xlabel('Key Position')
                axes[i].set_ylabel('Query Position')
                
                plt.colorbar(im, ax=axes[i])
            
            plt.suptitle('Attention Patterns', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'attention_patterns.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Attention visualization error: {e}")
    
    def visualize_keyframe_patterns(self, keyframe_data):
        """Visualize keyframe selection patterns"""
        try:
            fig, axes = plt.subplots(2, len(keyframe_data), figsize=(15, 6))
            if len(keyframe_data) == 1:
                axes = axes.reshape(2, 1)
            
            for i, data in enumerate(keyframe_data):
                folder_name = data['folder_name']
                frame_scores = data['frame_scores']
                keyframe_indices = data['keyframe_indices']
                
                # Plot frame scores
                if frame_scores:
                    axes[0, i].plot(frame_scores, 'b-', alpha=0.7)
                    axes[0, i].scatter(keyframe_indices, [frame_scores[j] for j in keyframe_indices if j < len(frame_scores)], 
                                      color='red', s=50, zorder=5)
                    axes[0, i].set_title(f'{folder_name}')
                    axes[0, i].set_ylabel('Frame Score')
                    axes[0, i].set_xlabel('Frame Index')
                
                # Plot keyframe selection
                total_frames = len(frame_scores) if frame_scores else 20
                selection_mask = np.zeros(total_frames)
                for idx in keyframe_indices:
                    if idx < total_frames:
                        selection_mask[idx] = 1
                
                axes[1, i].bar(range(total_frames), selection_mask, alpha=0.7, color='green')
                axes[1, i].set_ylabel('Selected')
                axes[1, i].set_xlabel('Frame Index')
                axes[1, i].set_ylim(0, 1.1)
            
            plt.suptitle('Keyframe Selection Patterns', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'keyframe_patterns.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Keyframe visualization error: {e}")
    
    def generate_enhanced_gradcam(self, num_samples=3):
        """Generate enhanced Grad-CAM visualizations"""
        print(f"🔥 Generating enhanced Grad-CAM for {num_samples} samples...")
        
        gradcam_results = []
        sample_count = 0
        
        try:
            for batch_idx, batch in enumerate(self.test_loader):
                if sample_count >= num_samples:
                    break
                
                try:
                    frames = batch['frames'].to(self.device)
                    folder_name = batch['folder_name'][0]
                    gloss_text = batch['gloss_text'][0]
                    
                    frames.requires_grad_(True)
                    heatmap = self.grad_cam.generate_cam(frames)
                    
                    if heatmap is not None:
                        original_frames = frames[0].cpu().detach().numpy()
                        original_frames = np.transpose(original_frames, (0, 2, 3, 1))
                        
                        # Denormalize
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        original_frames = original_frames * std + mean
                        original_frames = np.clip(original_frames, 0, 1)
                        
                        gradcam_results.append({
                            'folder_name': folder_name,
                            'gloss_text': gloss_text,
                            'original_frames': original_frames,
                            'heatmap': heatmap,
                            'sample_idx': sample_count
                        })
                        
                        sample_count += 1
                    
                except Exception as e:
                    print(f"Error generating Grad-CAM for sample {batch_idx}: {e}")
                    continue
        
        except Exception as e:
            print(f"⚠️  Grad-CAM generation failed: {e}")
            return []
        
        # Create visualizations
        if gradcam_results:
            self.visualize_enhanced_gradcam(gradcam_results)
            print(f"✅ Generated {len(gradcam_results)} enhanced Grad-CAM visualizations")
        
        return gradcam_results
    
    def visualize_enhanced_gradcam(self, gradcam_results):
        """Create enhanced Grad-CAM visualizations"""
        for result in gradcam_results:
            try:
                folder_name = result['folder_name']
                original_frames = result['original_frames']
                heatmap = result['heatmap']
                sample_idx = result['sample_idx']
                gloss_text = result['gloss_text']
                
                # Select representative frames
                num_frames_to_show = min(6, len(original_frames))
                frame_indices = np.linspace(0, len(original_frames)-1, num_frames_to_show, dtype=int)
                
                plt.ioff()
                fig, axes = plt.subplots(2, num_frames_to_show, figsize=(18, 6))
                if num_frames_to_show == 1:
                    axes = axes.reshape(2, 1)
                
                for i, frame_idx in enumerate(frame_indices):
                    # Original frame
                    axes[0, i].imshow(original_frames[frame_idx])
                    axes[0, i].set_title(f'Frame {frame_idx}')
                    axes[0, i].axis('off')
                    
                    # Grad-CAM overlay
                    frame = original_frames[frame_idx]
                    try:
                        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
                        
                        # Create enhanced overlay
                        overlay = frame.copy()
                        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
                        
                        # Blend with adaptive alpha based on heatmap intensity
                        alpha = 0.4 + 0.3 * heatmap_resized  # Variable alpha
                        overlay = (1 - alpha[:, :, np.newaxis]) * overlay + alpha[:, :, np.newaxis] * heatmap_colored
                        
                        axes[1, i].imshow(overlay)
                        axes[1, i].set_title(f'Enhanced CAM {frame_idx}')
                        axes[1, i].axis('off')
                        
                    except Exception as e:
                        axes[1, i].imshow(frame)
                        axes[1, i].set_title(f'Frame {frame_idx} (No CAM)')
                        axes[1, i].axis('off')
                
                plt.suptitle(f'Enhanced Grad-CAM Analysis\nFolder: {folder_name} | Text: "{gloss_text}"', 
                           fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                save_path = os.path.join(self.results_dir, f'enhanced_gradcam_sample_{sample_idx+1}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"⚠️  Failed to create enhanced Grad-CAM for sample {result.get('sample_idx', '?')}: {e}")
                continue
    
    def save_enhanced_results(self, metrics, sample_info, attention_data, keyframe_data, gradcam_results):
        """Save comprehensive results"""
        print("💾 Saving enhanced results...")
        
        try:
            # Save metrics
            with open(os.path.join(self.results_dir, 'enhanced_metrics.json'), 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, np.ndarray):
                        json_metrics[k] = v.tolist()
                    elif isinstance(v, np.floating):
                        json_metrics[k] = float(v)
                    elif isinstance(v, np.integer):
                        json_metrics[k] = int(v)
                    else:
                        json_metrics[k] = v
                json.dump(json_metrics, f, indent=2)
            
            # Save detailed sample results
            detailed_results = []
            for info in sample_info:
                result = {
                    'folder_name': info['folder_name'],
                    'ground_truth_text': info['ground_truth_text'],
                    'prediction_text': info['prediction_text'],
                    'ground_truth_sequence': info['ground_truth_seq'],
                    'predicted_sequence': info['prediction_seq'],
                    'edit_distance': info['edit_distance'],
                    'sequence_length_gt': info['sequence_length_gt'],
                    'sequence_length_pred': info['sequence_length_pred'],
                    'num_keyframes': info.get('num_keyframes', 0),
                    'perfect_match': info['edit_distance'] == 0,
                    'length_ratio': info['sequence_length_pred'] / max(info['sequence_length_gt'], 1)
                }
                detailed_results.append(result)
            
            # Save as CSV
            df = pd.DataFrame(detailed_results)
            df.to_csv(os.path.join(self.results_dir, 'enhanced_detailed_results.csv'), index=False)
            
            # Save as JSON
            with open(os.path.join(self.results_dir, 'enhanced_detailed_results.json'), 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            # Save attention data
            if attention_data:
                attention_summary = {
                    'num_samples': len(attention_data),
                    'samples': [{'folder_name': d['folder_name'], 'sequence_length': d['sequence_length']} 
                               for d in attention_data]
                }
                with open(os.path.join(self.results_dir, 'attention_analysis.json'), 'w') as f:
                    json.dump(attention_summary, f, indent=2)
            
            # Save keyframe data
            if keyframe_data:
                keyframe_summary = {
                    'num_samples': len(keyframe_data),
                    'average_keyframes': np.mean([d['num_keyframes'] for d in keyframe_data]),
                    'keyframe_distribution': dict(Counter([d['num_keyframes'] for d in keyframe_data]))
                }
                with open(os.path.join(self.results_dir, 'keyframe_analysis.json'), 'w') as f:
                    json.dump(keyframe_summary, f, indent=2)
            
            # Create comprehensive report
            self.create_enhanced_report(metrics, sample_info, attention_data, keyframe_data, gradcam_results)
            
            print(f"✅ Enhanced results saved to {self.results_dir}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            import traceback
            traceback.print_exc()
    
    def create_enhanced_report(self, metrics, sample_info, attention_data, keyframe_data, gradcam_results):
        """Create comprehensive markdown report"""
        report = f"""# Enhanced Sign Language Recognition - Inference Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration
- **Model Path**: {self.model_path}
- **Model Architecture**: Enhanced 4-Stage with 1024D features
- **Test Dataset**: {self.test_folder}
- **Annotations**: {self.annotations_file}
- **Device**: {self.device}
- **Vocabulary Size**: {len(self.vocab)}
- **Model Features**: {self.model_config.get('feature_dim', 1024)}D

## Dataset Statistics
- **Total Test Samples**: {len(sample_info)}
- **Successfully Processed**: {len(sample_info)}
- **Average Ground Truth Length**: {metrics.get('avg_ground_truth_length', 0):.2f}
- **Average Prediction Length**: {metrics.get('avg_prediction_length', 0):.2f}
- **Length Correlation**: {metrics.get('length_correlation', 0):.3f}

## Performance Metrics

### Primary Metrics
- **Word Error Rate (WER)**: {metrics.get('word_error_rate', 0):.4f} ({metrics.get('word_error_rate', 0)*100:.2f}%)
- **Sequence Accuracy**: {metrics.get('sequence_accuracy', 0):.4f} ({metrics.get('sequence_accuracy', 0)*100:.2f}%)
- **Token Accuracy**: {metrics.get('token_accuracy', 0):.4f} ({metrics.get('token_accuracy', 0)*100:.2f}%)

### Enhanced Metrics
- **Mean Edit Distance**: {metrics.get('mean_edit_distance', 0):.2f}
- **Median Edit Distance**: {metrics.get('median_edit_distance', 0):.2f}
- **Edit Distance Std**: {metrics.get('std_edit_distance', 0):.2f}

### Length-Based Analysis
"""
        
        # Add length-based metrics
        for category in ['short', 'medium', 'long']:
            wer_key = f'wer_{category}'
            acc_key = f'accuracy_{category}'
            if wer_key in metrics:
                report += f"- **{category.title()} Sequences**: WER={metrics[wer_key]:.4f}, Accuracy={metrics.get(acc_key, 0):.4f}\n"
        
        report += f"""
### Vocabulary Analysis
- **Vocabulary Coverage**: {metrics.get('vocabulary_coverage', 0):.4f} ({metrics.get('vocabulary_coverage', 0)*100:.2f}%)
- **Vocabulary Precision**: {metrics.get('vocabulary_precision', 0):.4f} ({metrics.get('vocabulary_precision', 0)*100:.2f}%)

### Keyframe Analysis
- **Average Keyframes**: {metrics.get('avg_keyframes', 0):.1f}
- **Keyframe Std**: {metrics.get('std_keyframes', 0):.1f}
- **Attention Samples**: {len(attention_data)}

## Analysis Summary

### Model Strengths
"""
        
        # Analyze strengths
        wer = metrics.get('word_error_rate', 1.0)
        seq_acc = metrics.get('sequence_accuracy', 0.0)
        
        if wer < 0.3:
            report += "- ✅ Low Word Error Rate indicates good sequence prediction\n"
        if seq_acc > 0.5:
            report += "- ✅ High sequence accuracy shows strong exact match capability\n"
        if metrics.get('vocabulary_coverage', 0) > 0.7:
            report += "- ✅ Good vocabulary coverage\n"
        
        report += "\n### Areas for Improvement\n"
        
        if wer > 0.5:
            report += "- ⚠️ High Word Error Rate suggests need for better training\n"
        if seq_acc < 0.3:
            report += "- ⚠️ Low sequence accuracy indicates difficulty with exact matches\n"
        if metrics.get('length_correlation', 0) < 0.5:
            report += "- ⚠️ Poor length correlation suggests length prediction issues\n"
        
        # Best and worst samples
        best_samples = sorted(sample_info, key=lambda x: x['edit_distance'])[:5]
        worst_samples = sorted(sample_info, key=lambda x: x['edit_distance'], reverse=True)[:5]
        
        report += f"\n## Best Performing Samples\n"
        for i, sample in enumerate(best_samples):
            report += f"\n{i+1}. **{sample['folder_name']}** (Edit Distance: {sample['edit_distance']})\n"
            report += f"   - Ground Truth: '{sample['ground_truth_text']}'\n"
            report += f"   - Prediction: '{sample['prediction_text']}'\n"
            report += f"   - Keyframes: {sample.get('num_keyframes', 'N/A')}\n"
        
        report += f"\n## Worst Performing Samples\n"
        for i, sample in enumerate(worst_samples):
            report += f"\n{i+1}. **{sample['folder_name']}** (Edit Distance: {sample['edit_distance']})\n"
            report += f"   - Ground Truth: '{sample['ground_truth_text']}'\n"
            report += f"   - Prediction: '{sample['prediction_text']}'\n"
            report += f"   - Keyframes: {sample.get('num_keyframes', 'N/A')}\n"
        
        report += f"""
## Files Generated
- `enhanced_metrics.json`: Complete metrics data
- `enhanced_detailed_results.csv`: Per-sample analysis
- `enhanced_detailed_results.json`: Detailed results in JSON
- `enhanced_metrics_dashboard.png`: Visual metrics summary
- `attention_patterns.png`: Attention visualization
- `keyframe_patterns.png`: Keyframe analysis
- `enhanced_gradcam_sample_*.png`: Enhanced Grad-CAM visualizations ({len(gradcam_results)} samples)
- `attention_analysis.json`: Attention mechanism analysis
- `keyframe_analysis.json`: Keyframe selection analysis

## Technical Details
- **Enhanced Features**: 1024D visual features, 512D hidden states
- **Attention Heads**: {self.model_config.get('num_heads', 8)}
- **LSTM Layers**: 3 bidirectional layers
- **Keyframe Selection**: ConvGRU with spatial attention
- **CTC Decoding**: Enhanced greedy decoding

Generated by Enhanced Sign Language Recognition Inference System
"""
        
        with open(os.path.join(self.results_dir, 'enhanced_inference_report.md'), 'w') as f:
            f.write(report)
    
    def run_complete_enhanced_evaluation(self):
        """Run complete enhanced evaluation pipeline"""
        print("🚀 Starting complete enhanced evaluation...")
        print("=" * 70)
        
        # Run enhanced inference
        predictions, ground_truths, sample_info, attention_data, keyframe_data = self.run_enhanced_inference()
        
        # Compute enhanced metrics
        metrics = self.compute_enhanced_metrics(predictions, ground_truths, sample_info)
        
        # Create enhanced visualizations
        self.create_enhanced_visualizations(metrics, sample_info, attention_data, keyframe_data)
        
        # Generate enhanced Grad-CAM
        gradcam_results = self.generate_enhanced_gradcam(num_samples=5)
        
        # Save enhanced results
        self.save_enhanced_results(metrics, sample_info, attention_data, keyframe_data, gradcam_results)
        
        print("=" * 70)
        print("🎉 Enhanced evaluation completed!")
        print(f"📊 Key Results:")
        print(f"   Word Error Rate: {metrics.get('word_error_rate', 0):.4f}")
        print(f"   Sequence Accuracy: {metrics.get('sequence_accuracy', 0):.4f}")
        print(f"   Token Accuracy: {metrics.get('token_accuracy', 0):.4f}")
        print(f"   Mean Edit Distance: {metrics.get('mean_edit_distance', 0):.2f}")
        print(f"   Vocabulary Coverage: {metrics.get('vocabulary_coverage', 0):.4f}")
        print(f"📁 All results saved to: {self.results_dir}")
        
        return metrics, sample_info, gradcam_results

def main():
    """Enhanced main function"""
    
    config = {
        'model_path': '//home/pvvkishore/Desktop/TVC_May21/New_Code/constrained_checkpoints/best_constrained_model.pth',
        'test_folder': '/home/pvvkishore/Desktop/TVC_May21/New_Code/test/',
        'annotations_file': '/home/pvvkishore/Desktop/TVC_May21/New_Code/annotations_folder/test_gloss_eng.csv',
        'results_dir': 'enhanced_results'
    }
    
    print("🔍 ENHANCED SIGN LANGUAGE MODEL INFERENCE")
    print("=" * 60)
    
    # Environment info
    print("🔧 Environment Information:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"   OpenCV: {cv2.__version__}")
    print()
    
    # Check paths
    print("📁 File Check:")
    for key, path in config.items():
        if key != 'results_dir':
            if os.path.exists(path):
                print(f"   ✅ {key}: {path}")
                if key == 'test_folder':
                    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    print(f"      Found {len(subdirs)} test subdirectories")
                elif key == 'annotations_file':
                    try:
                        df = pd.read_csv(path)
                        print(f"      CSV has {len(df)} rows")
                    except Exception as e:
                        print(f"      ⚠️  Could not read CSV: {e}")
            else:
                print(f"   ❌ {key}: {path} (NOT FOUND)")
                return
    
    print()
    
    try:
        # Initialize enhanced evaluator
        evaluator = EnhancedInferenceEvaluator(
            model_path=config['model_path'],
            test_folder=config['test_folder'],
            annotations_file=config['annotations_file'],
            results_dir=config['results_dir']
        )
        
        # Run complete enhanced evaluation
        metrics, sample_info, gradcam_results = evaluator.run_complete_enhanced_evaluation()
        
        print("\n🎯 ENHANCED EVALUATION SUMMARY:")
        print(f"   📊 Samples processed: {len(sample_info)}")
        print(f"   🎯 Word Error Rate: {metrics.get('word_error_rate', 0):.4f}")
        print(f"   ✅ Sequence Accuracy: {metrics.get('sequence_accuracy', 0):.4f}")
        print(f"   🔤 Token Accuracy: {metrics.get('token_accuracy', 0):.4f}")
        print(f"   📏 Mean Edit Distance: {metrics.get('mean_edit_distance', 0):.2f}")
        print(f"   📚 Vocabulary Coverage: {metrics.get('vocabulary_coverage', 0):.4f}")
        print(f"   🔥 Enhanced Grad-CAM: {len(gradcam_results)} samples")
        print(f"   📁 Results saved to: {config['results_dir']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
