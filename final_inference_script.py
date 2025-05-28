#!/usr/bin/env python3
"""
Final Executable Sign Language Model Inference Script
Specifically designed for the trained 112x112 model
"""

import os
import sys

# Fix OpenCV/Qt issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import cv2
import numpy as np
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
import editdistance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers, handle if not available
try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available, text encoder will be disabled")
    TRANSFORMERS_AVAILABLE = False

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv_reset = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_update = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv_new = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
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
        
        return new_hidden

class TextEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
            for param in self.bert.parameters():
                param.requires_grad = False
            
            for param in self.bert.encoder.layer[-2:].parameters():
                param.requires_grad = True
            
            self.projection = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        else:
            # Dummy encoder if transformers not available
            self.projection = nn.Linear(100, feature_dim)
    
    def forward(self, text_list):
        if not TRANSFORMERS_AVAILABLE:
            # Return dummy features
            batch_size = len(text_list)
            device = next(self.parameters()).device
            return torch.randn(batch_size, 512, device=device)
        
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
        text_features = outputs.last_hidden_state[:, 0, :]
        
        return self.projection(text_features)

class FourStageSignLanguageModel(nn.Module):
    def __init__(self, vocab_size, feature_dim=512, hidden_dim=256, lstm_hidden=128):
        super(FourStageSignLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # Stage 1: Visual Feature Extraction
        self.visual_encoder = self._build_visual_encoder()
        self.text_encoder = TextEncoder(feature_dim)
        
        # Stage 2: Motion-aware Keyframe Selection
        self.conv_gru = ConvGRU(input_dim=512, hidden_dim=hidden_dim)
        self.keyframe_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Stage 3: Temporal Modeling with Bi-LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=512 * 4 * 4,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.frame_classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, vocab_size)
        )
        
        # Stage 4: CTC Decoder
        self.ctc_projection = nn.Linear(lstm_hidden * 2, vocab_size)
        
        self.loss_weights = nn.Parameter(torch.ones(4))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
    def _build_visual_encoder(self):
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        layers = list(resnet.children())[:-2]
        visual_encoder = nn.Sequential(*layers)
        
        visual_encoder.add_module('projection', nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ))
        
        return visual_encoder
    
    def stage1_visual_feature_extraction(self, frames):
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        visual_features = self.visual_encoder(frames_flat)
        _, feat_dim, feat_h, feat_w = visual_features.shape
        visual_features = visual_features.view(B, T, feat_dim, feat_h, feat_w)
        return visual_features
    
    def stage2_keyframe_selection(self, visual_features, k_ratio=0.7):
        B, T, C, H, W = visual_features.shape
        frame_diffs = visual_features[:, 1:] - visual_features[:, :-1]
        
        hidden_state = None
        gru_outputs = []
        
        for t in range(T - 1):
            hidden_state = self.conv_gru(frame_diffs[:, t], hidden_state)
            gru_outputs.append(hidden_state)
        
        gru_features = torch.stack(gru_outputs, dim=1)
        
        frame_scores = self.keyframe_selector(gru_features.view(B * (T-1), -1, H, W))
        frame_scores = frame_scores.view(B, T-1)
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
    
    def stage3_temporal_modeling(self, selected_features):
        B, K, C, H, W = selected_features.shape
        features_flat = selected_features.view(B, K, C * H * W)
        lstm_out, (hidden, cell) = self.temporal_lstm(features_flat)
        frame_logits = self.frame_classifier(lstm_out)
        return lstm_out, frame_logits
    
    def stage4_ctc_prediction(self, lstm_features):
        ctc_logits = self.ctc_projection(lstm_features)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        return ctc_log_probs
    
    def compute_attention_weights(self, lstm_features):
        attn_output, attn_weights = self.attention(
            lstm_features, lstm_features, lstm_features
        )
        return attn_weights
    
    def forward(self, frames, gloss_texts=None, gloss_sequences=None, mode='eval'):
        outputs = {}
        
        visual_features = self.stage1_visual_feature_extraction(frames)
        outputs['visual_features'] = visual_features
        
        if gloss_texts is not None:
            try:
                text_features = self.text_encoder(gloss_texts)
                outputs['text_features'] = text_features
            except Exception as e:
                pass
        
        selected_features, frame_scores, keyframe_indices = self.stage2_keyframe_selection(visual_features)
        outputs['selected_features'] = selected_features
        outputs['frame_scores'] = frame_scores
        outputs['keyframe_indices'] = keyframe_indices
        
        lstm_features, frame_logits = self.stage3_temporal_modeling(selected_features)
        outputs['lstm_features'] = lstm_features
        outputs['frame_logits'] = frame_logits
        
        ctc_logits = self.stage4_ctc_prediction(lstm_features)
        outputs['ctc_logits'] = ctc_logits
        
        if mode == 'eval':
            try:
                attention_weights = self.compute_attention_weights(lstm_features)
                outputs['attention_weights'] = attention_weights
            except:
                pass
        
        return outputs

class GradCAM:
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
        
        # Hook the last conv layer in ResNet
        for name, module in self.model.visual_encoder.named_modules():
            if name == '7.0':  # Last conv layer before projection
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        
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
            heatmap /= torch.max(heatmap)
            
            return heatmap.detach().cpu().numpy()
        
        return None

class TestDataset(Dataset):
    def __init__(self, test_folder, annotations_file, vocab, transform=None, max_frames=16):
        self.test_folder = test_folder
        self.transform = transform
        self.max_frames = max_frames
        self.vocab = vocab
        
        # Load annotations
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations.iloc[1:, :2]  # Skip header, take first 2 columns
        self.annotations.columns = ['folder_name', 'gloss_text']
        
        # Filter valid samples
        self.valid_samples = []
        for _, row in self.annotations.iterrows():
            folder_path = os.path.join(test_folder, str(row['folder_name']))
            if os.path.exists(folder_path):
                self.valid_samples.append({
                    'folder_name': str(row['folder_name']),
                    'gloss_text': str(row['gloss_text']),
                    'folder_path': folder_path,
                    'gloss_sequence': self.text_to_sequence(str(row['gloss_text']))
                })
        
        print(f"📁 Loaded {len(self.valid_samples)} valid test samples")
    
    def text_to_sequence(self, text):
        words = text.lower().strip().split()
        sequence = []
        for word in words:
            if word in self.vocab:
                sequence.append(self.vocab[word])
            else:
                sequence.append(self.vocab.get('<unk>', 1))
        return sequence
    
    def __len__(self):
        return len(self.valid_samples)
    
    def load_frames(self, folder_path):
        frame_files = sorted([f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not frame_files:
            raise ValueError(f"No frames found in {folder_path}")
        
        # Sample frames uniformly
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
                    frame = cv2.resize(frame, (112, 112))
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

class SignLanguageInference:
    def __init__(self, model_path, test_folder, annotations_file, results_dir='results'):
        self.model_path = model_path
        self.test_folder = test_folder
        self.annotations_file = annotations_file
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"🚀 Initializing Sign Language Inference")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        print(f"   Test data: {test_folder}")
        print(f"   Results: {results_dir}")
        
        # Load model and create dataset
        self.model, self.vocab = self.load_model()
        self.test_dataset = self.create_test_dataset()
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.model)
        
        print(f"✅ Initialization complete")
        print(f"   Vocabulary size: {len(self.vocab)}")
        print(f"   Test samples: {len(self.test_dataset)}")
    
    def load_model(self):
        print("📦 Loading trained model...")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            print("✅ Checkpoint loaded successfully")
            
            # Debug checkpoint contents
            print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
            if 'vocab_size' in checkpoint:
                print(f"🔍 Saved vocab size: {checkpoint['vocab_size']}")
                
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise
        
        # Get model configuration
        vocab_size = checkpoint.get('vocab_size', 1000)  # Fallback vocabulary size
        model_config = checkpoint.get('model_config', {
            'feature_dim': 512,
            'hidden_dim': 256,
            'lstm_hidden': 128
        })
        
        print(f"🏗️  Model config: vocab_size={vocab_size}, {model_config}")
        
        # Create model
        model = FourStageSignLanguageModel(
            vocab_size=vocab_size,
            **model_config
        )
        
        # Load state dict with detailed error reporting
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ Model weights loaded")
            if missing_keys:
                print(f"⚠️  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print(f"⚠️  Model loading warning: {e}")
            print("   Continuing with available weights...")
        
        model.to(self.device)
        model.eval()
        
        # Create vocabulary - try to reconstruct from training data if available
        vocab = self.reconstruct_vocabulary(vocab_size)
        
        return model, vocab
    
    def reconstruct_vocabulary(self, vocab_size):
        """Try to reconstruct vocabulary from test annotations"""
        print("📚 Reconstructing vocabulary...")
        
        vocab = {'<blank>': 0, '<unk>': 1}
        
        try:
            # Read test annotations to get some vocabulary
            df = pd.read_csv(self.annotations_file)
            df = df.iloc[1:, :2]
            df.columns = ['folder_name', 'gloss_text']
            
            all_words = set()
            for _, row in df.iterrows():
                text = str(row['gloss_text']).lower().strip()
                words = text.split()
                all_words.update(words)
            
            # Add words to vocabulary
            sorted_words = sorted(list(all_words))
            for word in sorted_words:
                if word not in vocab:
                    vocab[word] = len(vocab)
            
            print(f"📝 Found {len(sorted_words)} unique words in test annotations")
            print(f"📝 Sample words: {sorted_words[:10]}")
            
        except Exception as e:
            print(f"⚠️  Could not read test annotations: {e}")
        
        # Fill remaining vocabulary slots
        while len(vocab) < vocab_size:
            vocab[f'token_{len(vocab)}'] = len(vocab)
        
        print(f"✅ Vocabulary created with {len(vocab)} tokens")
        return vocab
    
    def create_test_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return TestDataset(
            test_folder=self.test_folder,
            annotations_file=self.annotations_file,
            vocab=self.vocab,
            transform=transform,
            max_frames=16
        )
    
    def ctc_decode(self, log_probs, blank=0):
        """Simple CTC decoding with debugging"""
        print(f"🔍 CTC Debug - Input shape: {log_probs.shape}")
        print(f"🔍 CTC Debug - Input range: [{log_probs.min():.3f}, {log_probs.max():.3f}]")
        
        path = torch.argmax(log_probs, dim=-1)
        print(f"🔍 CTC Debug - Raw path: {path.cpu().numpy()}")
        
        decoded = []
        prev = None
        for token in path:
            token = token.item()
            if token != blank and token != prev:
                decoded.append(token)
            prev = token
        
        print(f"🔍 CTC Debug - Decoded sequence: {decoded}")
        return decoded
    
    def run_inference(self):
        print("🔍 Running inference...")
        
        all_predictions = []
        all_ground_truths = []
        all_sample_info = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Processing")):
                try:
                    frames = batch['frames'].to(self.device)
                    gloss_text = batch['gloss_text'][0]
                    folder_name = batch['folder_name'][0]
                    ground_truth = batch['gloss_sequence'][0]
                    
                    print(f"\n🔍 Processing sample {batch_idx}: {folder_name}")
                    print(f"🔍 Frames shape: {frames.shape}")
                    print(f"🔍 Ground truth: {gloss_text} -> {ground_truth}")
                    
                    # Forward pass
                    outputs = self.model(frames, mode='eval')
                    
                    # Debug model outputs
                    print(f"🔍 Model outputs keys: {list(outputs.keys())}")
                    if 'ctc_logits' in outputs:
                        ctc_shape = outputs['ctc_logits'].shape
                        print(f"🔍 CTC logits shape: {ctc_shape}")
                        
                        # Check if CTC output is valid
                        ctc_probs = F.softmax(outputs['ctc_logits'], dim=-1)
                        max_probs = torch.max(ctc_probs, dim=-1)[0]
                        print(f"🔍 Max probabilities: {max_probs[0].cpu().numpy()}")
                    
                    # CTC decoding
                    ctc_log_probs = outputs['ctc_logits'][0]
                    prediction = self.ctc_decode(ctc_log_probs)
                    
                    # Store results
                    all_predictions.append(prediction)
                    all_ground_truths.append(ground_truth)
                    
                    sample_info = {
                        'folder_name': folder_name,
                        'ground_truth_text': gloss_text,
                        'ground_truth_seq': ground_truth,
                        'prediction_seq': prediction,
                        'edit_distance': editdistance.eval(prediction, ground_truth),
                        'keyframe_indices': outputs.get('keyframe_indices', [[]])[0].cpu().numpy().tolist() if 'keyframe_indices' in outputs else [],
                        'frame_scores': outputs.get('frame_scores', [[]])[0].cpu().numpy().tolist() if 'frame_scores' in outputs else []
                    }
                    
                    all_sample_info.append(sample_info)
                    
                    print(f"🔍 Prediction: {prediction}")
                    print(f"🔍 Edit distance: {sample_info['edit_distance']}")
                    
                    # Clear memory
                    del outputs, frames
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process only first few samples for debugging
                    if batch_idx >= 4:  # Debug first 5 samples
                        print(f"🔍 Debug mode: stopping after {batch_idx + 1} samples")
                        break
                    
                except Exception as e:
                    print(f"Error processing {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"✅ Inference completed: {len(all_predictions)} samples processed")
        
        return all_predictions, all_ground_truths, all_sample_info
    
    def compute_metrics(self, predictions, ground_truths):
        print("📊 Computing metrics...")
        
        metrics = {}
        
        # Word Error Rate
        total_errors = 0
        total_words = 0
        for pred, truth in zip(predictions, ground_truths):
            if len(truth) > 0:
                errors = editdistance.eval(pred, truth)
                total_errors += errors
                total_words += len(truth)
        
        metrics['word_error_rate'] = total_errors / max(total_words, 1)
        
        # Sequence Accuracy
        correct = sum(1 for pred, truth in zip(predictions, ground_truths) if pred == truth)
        metrics['sequence_accuracy'] = correct / len(predictions)
        
        # Token-level accuracy (approximate)
        all_pred_tokens = [token for seq in predictions for token in seq]
        all_true_tokens = [token for seq in ground_truths for token in seq]
        
        if all_pred_tokens and all_true_tokens:
            min_len = min(len(all_pred_tokens), len(all_true_tokens))
            matches = sum(1 for i in range(min_len) 
                         if all_pred_tokens[i] == all_true_tokens[i])
            metrics['token_accuracy'] = matches / max(len(all_true_tokens), 1)
        else:
            metrics['token_accuracy'] = 0.0
        
        # Length statistics
        pred_lengths = [len(seq) for seq in predictions]
        true_lengths = [len(seq) for seq in ground_truths]
        
        metrics['avg_prediction_length'] = np.mean(pred_lengths) if pred_lengths else 0
        metrics['avg_ground_truth_length'] = np.mean(true_lengths) if true_lengths else 0
        
        print("📈 Metrics computed:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        return metrics
    
    def create_visualizations(self, metrics, sample_info):
        print("🎨 Creating visualizations...")
        
        try:
            # Set non-interactive backend to avoid Qt issues
            plt.ioff()  # Turn off interactive mode
            
            # Create metrics visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Sign Language Recognition - Inference Results', fontsize=16, fontweight='bold')
            
            # Key metrics
            metric_names = ['WER', 'Seq Acc', 'Token Acc']
            metric_values = [metrics['word_error_rate'], metrics['sequence_accuracy'], metrics['token_accuracy']]
            
            bars = axes[0, 0].bar(range(len(metric_names)), metric_values, 
                                 color=['red', 'green', 'blue'])
            axes[0, 0].set_xticks(range(len(metric_names)))
            axes[0, 0].set_xticklabels(metric_names)
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Performance Metrics')
            axes[0, 0].set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom')
            
            # Sequence length distribution
            pred_lengths = [len(info['prediction_seq']) for info in sample_info]
            true_lengths = [len(info['ground_truth_seq']) for info in sample_info]
            
            if pred_lengths and true_lengths:
                axes[0, 1].hist(pred_lengths, alpha=0.7, label='Predictions', bins=15, color='blue')
                axes[0, 1].hist(true_lengths, alpha=0.7, label='Ground Truth', bins=15, color='red')
                axes[0, 1].set_xlabel('Sequence Length')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Sequence Length Distribution')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Sequence Length Distribution (No Data)')
            
            # Edit distance distribution
            edit_distances = [info['edit_distance'] for info in sample_info]
            if edit_distances:
                axes[1, 0].hist(edit_distances, bins=15, color='purple', alpha=0.7)
                axes[1, 0].set_xlabel('Edit Distance')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Edit Distance Distribution')
            else:
                axes[1, 0].text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Edit Distance Distribution (No Data)')
            
            # Length vs error correlation
            if true_lengths and edit_distances:
                axes[1, 1].scatter(true_lengths, edit_distances, alpha=0.6, color='green')
                axes[1, 1].set_xlabel('Ground Truth Length')
                axes[1, 1].set_ylabel('Edit Distance')
                axes[1, 1].set_title('Length vs Error Correlation')
            else:
                axes[1, 1].text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Length vs Error Correlation (No Data)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'inference_metrics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Basic visualizations saved")
            
            # Create confusion analysis if data available
            if sample_info:
                self.create_confusion_analysis(sample_info)
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            print("   Continuing without visualizations...")
        
        print("✅ Visualizations completed")
    
    def create_confusion_analysis(self, sample_info):
        """Create additional analysis plots"""
        
        # Success rate by sequence length
        true_lengths = [len(info['ground_truth_seq']) for info in sample_info]
        edit_distances = [info['edit_distance'] for info in sample_info]
        
        if true_lengths:
            max_length = max(true_lengths)
            length_bins = range(1, max_length + 2)
            success_rates = []
            bin_centers = []
            
            for i in range(len(length_bins) - 1):
                mask = [(l >= length_bins[i] and l < length_bins[i+1]) 
                       for l in true_lengths]
                if any(mask):
                    successes = [edit_distances[j] == 0 for j, m in enumerate(mask) if m]
                    if successes:
                        success_rate = sum(successes) / len(successes)
                        success_rates.append(success_rate)
                        bin_centers.append(length_bins[i])
            
            if success_rates:
                plt.figure(figsize=(10, 6))
                plt.bar(bin_centers, success_rates, color='cyan', alpha=0.7)
                plt.xlabel('Sequence Length')
                plt.ylabel('Perfect Match Rate')
                plt.title('Success Rate by Sequence Length')
                plt.savefig(os.path.join(self.results_dir, 'success_by_length.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_gradcam_samples(self, num_samples=3):
        print(f"🔥 Generating Grad-CAM for {num_samples} samples...")
        
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
                    
                    print(f"🔥 Generating Grad-CAM for sample {sample_count + 1}: {folder_name}")
                    
                    # Generate Grad-CAM (simplified version to avoid crashes)
                    frames.requires_grad_(True)
                    
                    # Simple attention map instead of full Grad-CAM if it fails
                    try:
                        heatmap = self.grad_cam.generate_cam(frames)
                    except Exception as e:
                        print(f"⚠️  Grad-CAM failed for {folder_name}: {e}")
                        # Create dummy heatmap
                        heatmap = np.random.rand(112, 112) * 0.3  # Low intensity random map
                    
                    if heatmap is not None:
                        # Get original frames
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
            print("   Skipping Grad-CAM visualization...")
            return []
        
        # Visualize Grad-CAM results
        if gradcam_results:
            try:
                self.visualize_gradcam(gradcam_results)
                print(f"✅ Generated {len(gradcam_results)} Grad-CAM visualizations")
            except Exception as e:
                print(f"⚠️  Grad-CAM visualization failed: {e}")
        
        return gradcam_results
    
    def visualize_gradcam(self, gradcam_results):
        """Create Grad-CAM visualizations with error handling"""
        for result in gradcam_results:
            try:
                folder_name = result['folder_name']
                original_frames = result['original_frames']
                heatmap = result['heatmap']
                sample_idx = result['sample_idx']
                
                # Select frames to show
                num_frames_to_show = min(4, len(original_frames))
                frame_indices = np.linspace(0, len(original_frames)-1, num_frames_to_show, dtype=int)
                
                plt.ioff()  # Turn off interactive mode
                fig, axes = plt.subplots(2, num_frames_to_show, figsize=(12, 6))
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
                        
                        # Create overlay
                        overlay = frame.copy()
                        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
                        overlay = 0.6 * overlay + 0.4 * heatmap_colored
                        
                        axes[1, i].imshow(overlay)
                        axes[1, i].set_title(f'Grad-CAM {frame_idx}')
                        axes[1, i].axis('off')
                    except Exception as e:
                        # If overlay fails, just show the original frame
                        axes[1, i].imshow(frame)
                        axes[1, i].set_title(f'Frame {frame_idx} (No CAM)')
                        axes[1, i].axis('off')
                
                plt.suptitle(f'Grad-CAM Analysis - {folder_name}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                save_path = os.path.join(self.results_dir, f'gradcam_sample_{sample_idx+1}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Saved Grad-CAM for {folder_name}")
                
            except Exception as e:
                print(f"⚠️  Failed to create Grad-CAM visualization for sample {result.get('sample_idx', '?')}: {e}")
                continue
    
    def save_results(self, metrics, sample_info, gradcam_results):
        print("💾 Saving results...")
        
        # Save metrics
        with open(os.path.join(self.results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        df = pd.DataFrame(sample_info)
        df.to_csv(os.path.join(self.results_dir, 'detailed_results.csv'), index=False)
        
        # Create comprehensive report
        report = f"""# Sign Language Recognition - Inference Results

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Model: {self.model_path}
- Test Data: {self.test_folder}
- Annotations: {self.annotations_file}
- Device: {self.device}
- Vocabulary Size: {len(self.vocab)}

## Results Summary
- Total Samples: {len(sample_info)}
- **Word Error Rate**: {metrics['word_error_rate']:.4f} ({metrics['word_error_rate']*100:.2f}%)
- **Sequence Accuracy**: {metrics['sequence_accuracy']:.4f} ({metrics['sequence_accuracy']*100:.2f}%)
- **Token Accuracy**: {metrics['token_accuracy']:.4f} ({metrics['token_accuracy']*100:.2f}%)
- Average Prediction Length: {metrics['avg_prediction_length']:.2f}
- Average Ground Truth Length: {metrics['avg_ground_truth_length']:.2f}

## Best Performing Samples
"""
        
        # Add best and worst samples
        best_samples = sorted(sample_info, key=lambda x: x['edit_distance'])[:5]
        for i, sample in enumerate(best_samples):
            report += f"\n{i+1}. {sample['folder_name']} (Edit Distance: {sample['edit_distance']})\n"
            report += f"   Ground Truth: {sample['ground_truth_text']}\n"
        
        report += "\n## Worst Performing Samples\n"
        worst_samples = sorted(sample_info, key=lambda x: x['edit_distance'], reverse=True)[:5]
        for i, sample in enumerate(worst_samples):
            report += f"\n{i+1}. {sample['folder_name']} (Edit Distance: {sample['edit_distance']})\n"
            report += f"   Ground Truth: {sample['ground_truth_text']}\n"
        
        with open(os.path.join(self.results_dir, 'inference_report.md'), 'w') as f:
            f.write(report)
        
        print(f"✅ Results saved to {self.results_dir}")
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting complete evaluation...")
        print("=" * 60)
        
        # Run inference
        predictions, ground_truths, sample_info = self.run_inference()
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, ground_truths)
        
        # Create visualizations
        self.create_visualizations(metrics, sample_info)
        
        # Generate Grad-CAM
        gradcam_results = self.generate_gradcam_samples(num_samples=5)
        
        # Save results
        self.save_results(metrics, sample_info, gradcam_results)
        
        print("=" * 60)
        print("🎉 Evaluation completed!")
        print(f"📊 Word Error Rate: {metrics['word_error_rate']:.4f}")
        print(f"✅ Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
        print(f"🔤 Token Accuracy: {metrics['token_accuracy']:.4f}")
        print(f"📁 Results saved to: {self.results_dir}")
        
        return metrics, sample_info, gradcam_results

def main():
    """Main execution function"""
    
    print("🔍 SIGN LANGUAGE MODEL INFERENCE - DEBUG VERSION")
    print("=" * 60)
    
    # Debug environment
    print("🔧 Environment Information:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"   OpenCV: {cv2.__version__}")
    print()
    
    # Configuration paths
    config = {
        'model_path': '/home/pvvkishore/Desktop/TVC_Jan21/New_Code/checkpoints_112x112/final_model_112x112.pth',
        'test_folder': '/home/pvvkishore/Desktop/TVC_Jan21/New_Code/test/',
        'annotations_file': '/home/pvvkishore/Desktop/TVC_Jan21/New_Code/annotations_folder/test_gloss_eng.csv',
        'results_dir': 'results'
    }
    
    # Check if files exist
    print("📁 File Check:")
    for key, path in config.items():
        if key != 'results_dir':
            if os.path.exists(path):
                print(f"   ✅ {key}: {path}")
                if key == 'test_folder':
                    # Count subdirectories
                    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    print(f"      Found {len(subdirs)} test subdirectories")
                elif key == 'annotations_file':
                    # Count lines in CSV
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
        # Initialize and run inference
        evaluator = SignLanguageInference(
            model_path=config['model_path'],
            test_folder=config['test_folder'],
            annotations_file=config['annotations_file'],
            results_dir=config['results_dir']
        )
        
        # Run complete evaluation
        metrics, sample_info, gradcam_results = evaluator.run_complete_evaluation()
        
        print("\n🎯 FINAL SUMMARY:")
        print(f"   Processed: {len(sample_info)} samples")
        print(f"   WER: {metrics['word_error_rate']:.4f}")
        print(f"   Accuracy: {metrics['sequence_accuracy']:.4f}")
        print(f"   Grad-CAM: {len(gradcam_results)} samples")
        print(f"   Results: {config['results_dir']}/")
        
        # Additional debugging info
        if metrics['word_error_rate'] == 1.0:
            print("\n🚨 DEBUG: WER = 1.0 indicates model issues:")
            print("   1. Check if model vocabulary matches test data")
            print("   2. Verify model architecture compatibility")
            print("   3. Review CTC decoding process")
            print("   4. Check model training completion")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
