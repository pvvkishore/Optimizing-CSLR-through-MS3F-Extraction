# ===================================================================
# test_model.py - Comprehensive testing suite
import torch
import torch.nn.functional as F
import unittest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import json

# Import our modules
try:
    from model import FourStageSignLanguageModel, MultiStageLoss
    from data_loader import SignLanguageDataset, get_train_test_data_loaders
    from training import FourStageTrainer
    from evaluation import ComprehensiveEvaluator
    from visualizations import SignLanguageVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the same directory")

class TestFourStageModel(unittest.TestCase):
    """Test cases for the 4-stage sign language model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vocab_size = 100
        self.batch_size = 2
        self.num_frames = 16
        self.device = 'cpu'  # Use CPU for testing
        
        # Create model
        self.model = FourStageSignLanguageModel(
            vocab_size=self.vocab_size,
            feature_dim=256,  # Smaller for testing
            hidden_dim=128,
            lstm_hidden=64
        )
        self.model.to(self.device)
        
        # Create dummy data
        self.dummy_frames = torch.randn(self.batch_size, self.num_frames, 3, 224, 224)
        self.dummy_texts = ["hello world", "goodbye friend"]
        self.dummy_sequences = [[1, 2, 3], [4, 5]]
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, FourStageSignLanguageModel)
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        
        # Check if all components are initialized
        self.assertIsNotNone(self.model.visual_encoder)
        self.assertIsNotNone(self.model.text_encoder)
        self.assertIsNotNone(self.model.conv_gru)
        self.assertIsNotNone(self.model.temporal_lstm)
        self.assertIsNotNone(self.model.ctc_projection)
        
    def test_stage1_visual_features(self):
        """Test Stage 1: Visual feature extraction"""
        visual_features = self.model.stage1_visual_feature_extraction(self.dummy_frames)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_frames, 512, 4, 4)
        self.assertEqual(visual_features.shape, expected_shape)
        
        # Check if features are not all zeros
        self.assertFalse(torch.allclose(visual_features, torch.zeros_like(visual_features)))
        
    def test_stage2_keyframe_selection(self):
        """Test Stage 2: Keyframe selection"""
        visual_features = self.model.stage1_visual_feature_extraction(self.dummy_frames)
        selected_features, frame_scores, keyframe_indices = self.model.stage2_keyframe_selection(visual_features)
        
        # Check shapes
        expected_k = max(1, int((self.num_frames - 1) * 0.7))
        self.assertEqual(selected_features.shape[0], self.batch_size)
        self.assertEqual(selected_features.shape[1], expected_k)
        self.assertEqual(frame_scores.shape, (self.batch_size, self.num_frames - 1))
        self.assertEqual(keyframe_indices.shape, (self.batch_size, expected_k))
        
        # Check if frame scores sum to 1 (softmax)
        score_sums = torch.sum(frame_scores, dim=1)
        self.assertTrue(torch.allclose(score_sums, torch.ones(self.batch_size), atol=1e-5))
        
    def test_stage3_temporal_modeling(self):
        """Test Stage 3: Temporal modeling"""
        visual_features = self.model.stage1_visual_feature_extraction(self.dummy_frames)
        selected_features, _, _ = self.model.stage2_keyframe_selection(visual_features)
        lstm_features, frame_logits = self.model.stage3_temporal_modeling(selected_features)
        
        # Check shapes
        k = selected_features.shape[1]
        expected_lstm_shape = (self.batch_size, k, 128)  # 2 * lstm_hidden
        expected_logits_shape = (self.batch_size, k, self.vocab_size)
        
        self.assertEqual(lstm_features.shape, expected_lstm_shape)
        self.assertEqual(frame_logits.shape, expected_logits_shape)
        
    def test_stage4_ctc_prediction(self):
        """Test Stage 4: CTC prediction"""
        visual_features = self.model.stage1_visual_feature_extraction(self.dummy_frames)
        selected_features, _, _ = self.model.stage2_keyframe_selection(visual_features)
        lstm_features, _ = self.model.stage3_temporal_modeling(selected_features)
        ctc_logits = self.model.stage4_ctc_prediction(lstm_features)
        
        # Check shapes
        k = selected_features.shape[1]
        expected_shape = (self.batch_size, k, self.vocab_size)
        self.assertEqual(ctc_logits.shape, expected_shape)
        
        # Check if output is log probabilities (should sum to 1 after softmax)
        probs = torch.softmax(ctc_logits, dim=-1)
        prob_sums = torch.sum(probs, dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))
        
    def test_full_forward_pass(self):
        """Test complete forward pass"""
        outputs = self.model(
            self.dummy_frames, 
            self.dummy_texts, 
            self.dummy_sequences, 
            mode='train'
        )
        
        # Check if all expected outputs are present
        expected_keys = [
            'visual_features', 'text_features', 'selected_features',
            'frame_scores', 'keyframe_indices', 'lstm_features',
            'frame_logits', 'ctc_logits'
        ]
        
        for key in expected_keys:
            self.assertIn(key, outputs)
            self.assertIsInstance(outputs[key], torch.Tensor)
        
    def test_loss_computation(self):
        """Test multi-stage loss computation"""
        criterion = MultiStageLoss()
        
        outputs = self.model(self.dummy_frames, self.dummy_texts, self.dummy_sequences)
        
        # Prepare dummy targets
        targets = {
            'gloss_sequences': self.dummy_sequences,
            'frame_labels': torch.randint(0, self.vocab_size, (self.batch_size, 10))
        }
        
        losses = criterion(outputs, targets)
        
        # Check if losses are computed
        self.assertIn('cross_modal', losses)
        self.assertIn('ctc', losses)
        
        # Check if losses are scalar tensors
        for loss_name, loss_value in losses.items():
            self.assertTrue(loss_value.dim() == 0)  # Scalar
            self.assertFalse(torch.isnan(loss_value))
            self.assertFalse(torch.isinf(loss_value))

class TestDataLoader(unittest.TestCase):
    """Test cases for data loading pipeline"""
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.create_dummy_dataset()
        
    def tearDown(self):
        """Clean up test data"""
        shutil.rmtree(self.temp_dir)
        
    def create_dummy_dataset(self):
        """Create dummy dataset for testing"""
        # Create root folder structure
        train_folder = os.path.join(self.temp_dir, 'train')
        annotations_folder = os.path.join(self.temp_dir, 'annotations')
        
        os.makedirs(train_folder)
        os.makedirs(annotations_folder)
        
        # Create dummy video folders with frames
        for i in range(5):
            video_folder = os.path.join(train_folder, f'video_{i:03d}')
            os.makedirs(video_folder)
            
            # Create dummy frames
            for j in range(10):
                frame_path = os.path.join(video_folder, f'frame_{j:03d}.jpg')
                # Create a small dummy image
                dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                from PIL import Image
                Image.fromarray(dummy_image).save(frame_path)
        
        # Create dummy CSV file
        csv_path = os.path.join(annotations_folder, 'train_gloss_eng.csv')
        with open(csv_path, 'w') as f:
            f.write("folder,gloss\n")  # Header
            f.write("header_row,header_gloss\n")  # Will be skipped
            for i in range(5):
                f.write(f"video_{i:03d},gloss_{i}\n")
        
        self.train_folder = train_folder
        self.annotations_folder = annotations_folder
        
    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = SignLanguageDataset(
            root_train_folder=self.train_folder,
            annotations_folder=self.annotations_folder,
            max_frames=8
        )
        
        self.assertEqual(len(dataset), 5)
        self.assertIsInstance(dataset.vocab, dict)
        self.assertGreater(len(dataset.vocab), 0)
        
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        dataset = SignLanguageDataset(
            root_train_folder=self.train_folder,
            annotations_folder=self.annotations_folder,
            max_frames=8
        )
        
        sample = dataset[0]
        
        # Check sample structure
        self.assertIn('frames', sample)
        self.assertIn('gloss_text', sample)
        self.assertIn('folder_name', sample)
        self.assertIn('gloss_sequence', sample)
        
        # Check frame shape
        self.assertEqual(sample['frames'].shape, (8, 3, 224, 224))
        
    def test_data_loader_creation(self):
        """Test data loader creation"""
        try:
            train_loader, test_loader, dataset = get_train_test_data_loaders(
                root_train_folder=self.train_folder,
                annotations_folder=self.annotations_folder,
                batch_size=2,
                test_split=0.4,  # High split for small dataset
                num_workers=0
            )
            
            self.assertGreater(len(train_loader), 0)
            self.assertGreater(len(test_loader), 0)
            
            # Test batch loading
            batch = next(iter(train_loader))
            self.assertIn('frames', batch)
            self.assertIn('gloss_text', batch)
            
        except Exception as e:
            self.fail(f"Data loader creation failed: {e}")

class TestTraining(unittest.TestCase):
    """Test cases for training pipeline"""
    
    def setUp(self):
        """Set up training test"""
        self.vocab_size = 50
        self.model = FourStageSignLanguageModel(
            vocab_size=self.vocab_size,
            feature_dim=128,
            hidden_dim=64,
            lstm_hidden=32
        )
        
        # Create dummy data loaders
        self.train_loader = self.create_dummy_loader()
        self.val_loader = self.create_dummy_loader()
        
    def create_dummy_loader(self):
        """Create dummy data loader for testing"""
        class DummyDataset:
            def __len__(self):
                return 4
            
            def __getitem__(self, idx):
                return {
                    'frames': torch.randn(16, 3, 224, 224),
                    'gloss_text': f'test gloss {idx}',
                    'folder_name': f'folder_{idx}',
                    'gloss_sequence': [1, 2, 3]
                }
        
        from torch.utils.data import DataLoader
        from data_loader import custom_collate_fn
        
        return DataLoader(
            DummyDataset(),
            batch_size=2,
            collate_fn=custom_collate_fn
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = FourStageTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            vocab_size=self.vocab_size,
            device='cpu'
        )
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)
        self.assertIsNotNone(trainer.criterion)
        
    def test_single_training_step(self):
        """Test single training step"""
        trainer = FourStageTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            vocab_size=self.vocab_size,
            device='cpu'
        )
        
        # Get a batch
        batch = next(iter(self.train_loader))
        
        try:
            # Test target preparation
            targets = trainer.prepare_targets(batch)
            self.assertIn('gloss_sequences', targets)
            self.assertIn('frame_labels', targets)
            
            # Test forward pass
            outputs = trainer.model(
                batch['frames'],
                batch['gloss_text'],
                batch['gloss_sequence']
            )
            
            # Test loss computation
            losses = trainer.criterion(outputs, targets)
            self.assertIsInstance(losses, dict)
            
        except Exception as e:
            self.fail(f"Training step failed: {e}")

class TestVisualization(unittest.TestCase):
    """Test cases for visualization functions"""
    
    def setUp(self):
        """Set up visualization test"""
        self.model = FourStageSignLanguageModel(vocab_size=50)
        self.visualizer = SignLanguageVisualizer(self.model, device='cpu')
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
        
    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        self.assertIsInstance(self.visualizer, SignLanguageVisualizer)
        self.assertEqual(self.visualizer.device, 'cpu')
        
    def test_feature_visualization(self):
        """Test feature visualization functions"""
        # Create dummy features
        visual_features = torch.randn(4, 256)
        text_features = torch.randn(4, 256)
        text_list = ['test1', 'test2', 'test3', 'test4']
        
        try:
            # Test similarity heatmap
            self.visualizer.plot_similarity_heatmap(
                visual_features, text_features, text_list, 
                save_dir=self.temp_dir
            )
            
            # Check if file was created
            heatmap_file = os.path.join(self.temp_dir, 'similarity_heatmap.png')
            self.assertTrue(os.path.exists(heatmap_file))
            
            # Test feature distribution
            self.visualizer.plot_feature_distribution(
                visual_features, text_features,
                save_dir=self.temp_dir
            )
            
            # Check if file was created
            dist_file = os.path.join(self.temp_dir, 'feature_distributions.png')
            self.assertTrue(os.path.exists(dist_file))
            
        except Exception as e:
            self.fail(f"Visualization failed: {e}")

class TestUtilities(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_config_creation(self):
        """Test configuration creation"""
        from utils import create_training_config
        
        config = create_training_config(
            root_folder='/test/root',
            annotations_folder='/test/annotations'
        )
        
        self.assertIsInstance(config, dict)
        self.assertIn('root_train_folder', config)
        self.assertIn('batch_size', config)
        self.assertIn('num_epochs', config)
        
    def test_environment_setup(self):
        """Test environment setup"""
        from utils import setup_environment
        
        try:
            setup_environment()
        except Exception as e:
            self.fail(f"Environment setup failed: {e}")

# ===================================================================
# config_loader.py - Configuration management
import yaml
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class DatasetConfig:
    root_train_folder: str
    annotations_folder: str
    csv_filename: str = "train_gloss_eng.csv"
    max_frames: int = 32
    image_size: tuple = (224, 224)
    test_split: float = 0.2
    validation_split: float = 0.1

@dataclass
class ModelConfig:
    feature_dim: int = 1024
    hidden_dim: int = 512
    lstm_hidden: int = 256
    vocab_size: Optional[int] = None
    keyframe_ratio: float = 0.7
    attention_heads: int = 8
    dropout: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 100
    learning_rates: Dict[str, float] = None
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = {
                'visual_encoder': 1.0e-4,
                'text_encoder': 2.0e-5,
                'conv_gru': 5.0e-4,
                'temporal_lstm': 3.0e-4,
                'frame_classifier': 3.0e-4,
                'ctc_projection': 3.0e-4,
                'loss_weights': 1.0e-3
            }

@dataclass
class EvaluationConfig:
    metrics: list = None
    eval_frequency: int = 5
    save_best_model: bool = True
    save_frequency: int = 10
    visualize_attention: bool = True
    visualize_keyframes: bool = True
    save_sample_predictions: bool = True
    num_visualization_samples: int = 10
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "word_error_rate",
                "sequence_accuracy", 
                "frame_accuracy",
                "cross_modal_similarity",
                "keyframe_selection_quality",
                "attention_analysis"
            ]

@dataclass
class HardwareConfig:
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False

@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "training.log"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "sign-language-recognition"

@dataclass
class DirectoriesConfig:
    checkpoints: str = "checkpoints"
    results: str = "evaluation_results"
    visualizations: str = "visualizations"  
    logs: str = "logs"

@dataclass
class ReproducibilityConfig:
    random_seed: int = 42
    deterministic: bool = True

@dataclass
class CompleteConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    directories: DirectoriesConfig
    reproducibility: ReproducibilityConfig

class ConfigLoader:
    """Configuration loading and management utility"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> CompleteConfig:
        """Load configuration from YAML file"""
        
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return self._dict_to_config(config_dict)
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._create_default_config()
    
    def save_config(self, config: CompleteConfig, path: Optional[str] = None):
        """Save configuration to YAML file"""
        save_path = path or self.config_path
        
        config_dict = asdict(config)
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> CompleteConfig:
        """Convert dictionary to configuration dataclass"""
        
        # Extract nested configurations
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        hardware_config = HardwareConfig(**config_dict.get('hardware', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        directories_config = DirectoriesConfig(**config_dict.get('directories', {}))
        reproducibility_config = ReproducibilityConfig(**config_dict.get('reproducibility', {}))
        
        return CompleteConfig(
            dataset=dataset_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            hardware=hardware_config,
            logging=logging_config,
            directories=directories_config,
            reproducibility=reproducibility_config
        )
    
    def _create_default_config(self) -> CompleteConfig:
        """Create default configuration"""
        return CompleteConfig(
            dataset=DatasetConfig(
                root_train_folder="/path/to/train",
                annotations_folder="/path/to/annotations"
            ),
            model=ModelConfig(),
            training=TrainingConfig(),
            evaluation=EvaluationConfig(),
            hardware=HardwareConfig(),
            logging=LoggingConfig(),
            directories=DirectoriesConfig(),
            reproducibility=ReproducibilityConfig()
        )
    
    def validate_config(self, config: CompleteConfig) -> bool:
        """Validate configuration values"""
        try:
            # Check dataset paths
            if not os.path.exists(config.dataset.root_train_folder):
                self.logger.error(f"Train folder not found: {config.dataset.root_train_folder}")
                return False
            
            if not os.path.exists(config.dataset.annotations_folder):
                self.logger.error(f"Annotations folder not found: {config.dataset.annotations_folder}")
                return False
            
            # Check model parameters
            if config.model.feature_dim <= 0:
                self.logger.error("Feature dimension must be positive")
                return False
            
            # Check training parameters
            if config.training.batch_size <= 0:
                self.logger.error("Batch size must be positive")
                return False
            
            if config.training.num_epochs <= 0:
                self.logger.error("Number of epochs must be positive")
                return False
            
            # Check splits
            if not 0 < config.dataset.test_split < 1:
                self.logger.error("Test split must be between 0 and 1")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

# ===================================================================
# logger_setup.py - Logging configuration
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_filename: Optional[str] = None,
    log_dir: str = "logs",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_filename: Custom log filename
        log_dir: Directory for log files
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    
    # Create log directory
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if log_to_file:
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"sign_language_training_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_filename)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    # Set specific loggers to appropriate levels
    # Reduce noise from external libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logger

def setup_tensorboard_logging(log_dir: str = "logs/tensorboard"):
    """Set up TensorBoard logging"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_dir = os.path.join(log_dir, f"run_{timestamp}")
        
        writer = SummaryWriter(tensorboard_dir)
        
        logging.getLogger(__name__).info(f"TensorBoard logging to: {tensorboard_dir}")
        return writer
        
    except ImportError:
        logging.getLogger(__name__).warning("TensorBoard not available")
        return None

def setup_wandb_logging(
    wandb_project: str = "sign-language-recognition",
    wandb_config: Optional[dict] = None,
    wandb_name: Optional[str] = None
):
    """Set up Weights & Biases logging"""
    try:
        import wandb
        
        if wandb_name is None:
            wandb_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=wandb_config
        )
        
        logging.getLogger(__name__).info(f"W&B logging initialized: {wandb_project}/{wandb_name}")
        return wandb
        
    except ImportError:
        logging.getLogger(__name__).warning("Weights & Biases not available")
        return None

# ===================================================================
# run_tests.py - Test runner script
import unittest
import sys
import os

def run_all_tests():
    """Run all test cases"""
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('.', pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

# ===================================================================
# benchmark.py - Performance benchmarking
import time
import torch
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from model import FourStageSignLanguageModel

class PerformanceBenchmark:
    """Benchmark performance of the sign language model"""
    
    def __init__(self, model: FourStageSignLanguageModel, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.results = {}
        
    def benchmark_inference_speed(self, batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict:
        """Benchmark inference speed for different batch sizes"""
        print("Benchmarking inference speed...")
        
        inference_times = {}
        memory_usage = {}
        
        self.model.eval()
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create dummy input
            frames = torch.randn(batch_size, 32, 3, 224, 224).to(self.device)
            texts = [f"test gloss {i}" for i in range(batch_size)]
            
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    _ = self.model(frames, texts, mode='eval')
            
            # Benchmark
            times = []
            memory_before = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
            
            for _ in range(20):
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = self.model(frames, texts, mode='eval')
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end_time = time.time()
                
                times.append(end_time - start_time)
            
            memory_after = torch.cuda.memory_allocated() if self.device == 'cuda' else 0
            
            inference_times[batch_size] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }
            
            memory_usage[batch_size] = memory_after - memory_before
        
        self.results['inference_speed'] = inference_times
        self.results['memory_usage'] = memory_usage
        
        return {'inference_times': inference_times, 'memory_usage': memory_usage}
    
    def benchmark_training_speed(self, batch_size: int = 4, num_steps: int = 10) -> Dict:
        """Benchmark training speed"""
        print("Benchmarking training speed...")
        
        self.model.train()
        
        # Create dummy data
        frames = torch.randn(batch_size, 32, 3, 224, 224).to(self.device)
        texts = [f"training gloss {i}" for i in range(batch_size)]
        sequences = [[1, 2, 3] for _ in range(batch_size)]
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Warm up
        for _ in range(3):
            outputs = self.model(frames, texts, sequences)
            # Dummy loss computation
            loss = torch.mean(outputs['ctc_logits'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Benchmark
        times = []
        
        for step in range(num_steps):
            start_time = time.time()
            
            outputs = self.model(frames, texts, sequences)
            loss = torch.mean(outputs['ctc_logits'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"Step {step+1}/{num_steps}: {end_time - start_time:.4f}s")
        
        training_speed = {
            'mean_step_time': np.mean(times),
            'std_step_time': np.std(times),
            'steps_per_second': 1.0 / np.mean(times)
        }
        
        self.results['training_speed'] = training_speed
        return training_speed
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage patterns"""
        print("Benchmarking memory usage...")
        
        memory_stats = {}
        
        if self.device == 'cuda':
            # GPU memory
            torch.cuda.empty_cache()
            memory_stats['gpu_memory'] = {
                'allocated_before': torch.cuda.memory_allocated(),
                'reserved_before': torch.cuda.memory_reserved(),
            }
            
            # Run inference
            frames = torch.randn(4, 32, 3, 224, 224).to(self.device)
            texts = ["test gloss"] * 4
            
            with torch.no_grad():
                outputs = self.model(frames, texts, mode='eval')
            
            memory_stats['gpu_memory'].update({
                'allocated_after': torch.cuda.memory_allocated(),
                'reserved_after': torch.cuda.memory_reserved(),
                'peak_allocated': torch.cuda.max_memory_allocated(),
                'peak_reserved': torch.cuda.max_memory_reserved()
            })
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_stats['cpu_memory'] = {
            'rss': memory_info.rss,  # Resident Set Size
            'vms': memory_info.vms,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
        
        self.results['memory_stats'] = memory_stats
        return memory_stats
    
    def generate_report(self, save_path: str = "benchmark_report.json"):
        """Generate comprehensive benchmark report"""
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'results': self.results
        }
        
        # Add system info
        if self.device == 'cuda':
            try:
                gpus = GPUtil.getGPUs()
                report['gpu_info'] = {
                    'name': gpus[0].name if gpus else 'Unknown',
                    'memory_total': gpus[0].memoryTotal if gpus else 0,
                    'driver_version': gpus[0].driver if gpus else 'Unknown'
                }
            except:
                report['gpu_info'] = 'GPU info not available'
        
        report['cpu_info'] = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Benchmark report saved to: {save_path}")
        return report
    
    def plot_results(self, save_dir: str = "benchmark_plots"):
        """Generate visualization plots for benchmark results"""
        os.makedirs(save_dir, exist_ok=True)
        
        if 'inference_speed' in self.results:
            self._plot_inference_speed(save_dir)
        
        if 'memory_usage' in self.results:
            self._plot_memory_usage(save_dir)
    
    def _plot_inference_speed(self, save_dir: str):
        """Plot inference speed results"""
        inference_times = self.results['inference_speed']
        
        batch_sizes = list(inference_times.keys())
        mean_times = [inference_times[bs]['mean'] for bs in batch_sizes]
        std_times = [inference_times[bs]['std'] for bs in batch_sizes]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(batch_sizes, mean_times, yerr=std_times, 
                    marker='o', capsize=5, capthick=2)
        plt.xlabel('Batch Size')
        plt.ylabel('Inference Time (seconds)')
        plt.title('Inference Speed vs Batch Size')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_speed.png'), dpi=150)
        plt.close()
    
    def _plot_memory_usage(self, save_dir: str):
        """Plot memory usage results"""
        memory_usage = self.results['memory_usage']
        
        batch_sizes = list(memory_usage.keys())
        memory_mb = [memory_usage[bs] / (1024 ** 2) for bs in batch_sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, memory_mb, marker='s', linewidth=2, markersize=8)
        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Batch Size')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'memory_usage.png'), dpi=150)
        plt.close()

def run_full_benchmark():
    """Run complete benchmark suite"""
    print("Starting comprehensive benchmark...")
    
    # Initialize model
    model = FourStageSignLanguageModel(vocab_size=1000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    benchmark = PerformanceBenchmark(model, device)
    
    # Run benchmarks
    benchmark.benchmark_inference_speed()
    benchmark.benchmark_training_speed()
    benchmark.benchmark_memory_usage()
    
    # Generate reports
    report = benchmark.generate_report()
    benchmark.plot_results()
    
    print("Benchmark completed!")
    return report

if __name__ == "__main__":
    # Run individual tests
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running unit tests...")
        run_all_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("Running performance benchmark...")
        run_full_benchmark()
    else:
        print("Usage:")
        print("  python testing_and_utilities.py test      # Run unit tests")
        print("  python testing_and_utilities.py benchmark # Run performance benchmark")
        print("  python testing_and_utilities.py           # Show this help")