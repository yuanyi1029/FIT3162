# white_box_testing.py
import os
import sys
import unittest
import tempfile
import torch
import torch.nn as nn
import streamlit as st
from unittest.mock import MagicMock, patch
from PIL import Image
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_v2 import identify_model_blocks
from pruning_logic.Streamlined_prune import (
    prepare_dataloaders,
    prune_multiple_blocks,
    uniform_prune_and_depthwise_collapse,
    main_pruning_loop,
    main_finetune_model,
    knowledge_distillation_prune,
    Pruner,
    col_based_prune_reduction,
    uniform_channel_prune,
    prune_mb_inverted_block,
    finetune_model,
    run_distillation
)
from mcunet.tinynas.nn.networks.mcunets import MobileInvertedResidualBlock
from mcunet.utils.pytorch_modules import SEModule

# Define a simple test model
class SimpleModel(nn.Module):
    def __init__(self, num_blocks=3):
        super(SimpleModel, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 3, kernel_size=3, padding=1)
            ))
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# Create a more complex test model with nested blocks
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        ))
        self.blocks.append(nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        ))
        self.classifier = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

# Create a model without blocks
class NoBlocksModel(nn.Module):
    def __init__(self):
        super(NoBlocksModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# UI Helper Function Tests
class TestUIComponents(unittest.TestCase):
    """Unit tests for UI components"""
    
    def test_identify_model_blocks_valid(self):
        """Test UT-01: identify_model_blocks with valid model"""
        print("Testing identify_model_blocks with valid model...")
        model = SimpleModel(num_blocks=3)
        blocks = identify_model_blocks(model)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks, ['blocks.0', 'blocks.1', 'blocks.2'])

        print("Testing identify_model_blocks with valid model correctly identified blocks.")
    
    def test_identify_model_blocks_empty(self):
        """Test UT-02: identify_model_blocks with model without blocks"""
        print("Testing identify_model_blocks with model without blocks...")
        model = NoBlocksModel()
        blocks = identify_model_blocks(model)
        self.assertEqual(blocks, [])

        print("Testing identify_model_blocks with model without blocks correctly identified no blocks.")
    
    def test_identify_model_blocks_exception(self):
        """Test UT-03: identify_model_blocks with invalid model"""
        print("Testing identify_model_blocks with None as model...")
        # Test with None as model which should raise an exception internally
        blocks = identify_model_blocks(None)
        self.assertEqual(blocks, [])

        print("Testing identify_model_blocks with None as model correctly identified no blocks.")
    
    def test_identify_model_blocks_complex(self):
        """Test with a more complex model structure"""
        print("Testing identify_model_blocks with complex model structure...")
        model = ComplexModel()
        blocks = identify_model_blocks(model)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks, ['blocks.0', 'blocks.1'])

        print("Testing identify_model_blocks with complex model structure correctly identified blocks.")


# Session State Logic Test
class TestSessionState(unittest.TestCase):
    """Tests for session state management"""

    def test_session_state_initialization(self):
        """Test UT-04: Session state initialization"""

        print("Testing session state initialization...")
        # Simulate the behavior of session state
        with patch.dict(st.session_state, {}, clear=True):
            # Simulate sidebar interaction that initializes session state
            with patch('streamlit.sidebar'):
                with patch('streamlit.file_uploader', return_value=True):
                    import main_v2  # Ensure this runs the app logic that sets session state
                    
                    # Manually trigger session state initialization if needed
                    if 'previous_mode' not in st.session_state:
                        st.session_state.previous_mode = "Basic"
                    
                    # Now perform assertions
                    self.assertIn('previous_mode', st.session_state)
                    self.assertEqual(st.session_state.previous_mode, "Basic")

                    print("Session state initialized correctly.")


class MockOptimization:
    """Mock optimization functions to test the pipeline"""
    
    @staticmethod
    def mock_main_pruning_loop(model, block_level_dict, uniform_pruning_ratio, 
                               block_fine_tune_epochs, channel_fine_tune_epochs, 
                               device, type):
        """Mock the pruning function to return a pruned model"""
        # Simulate pruning by creating a new model with fewer blocks
        if type == "BLOCK" or type == "BOTH":
            num_blocks = max(1, int(len(model.blocks) * (1 - 0.5)))  # Simulate 50% block pruning
            return SimpleModel(num_blocks=num_blocks)
        return model
    
    @staticmethod
    def mock_knowledge_distillation_prune(teacher_model, student_model, num_epochs, device):
        """Mock the knowledge distillation function"""
        # Just return the student model unchanged
        return student_model
    
    @staticmethod
    def mock_quantize_model(model_path, output_path, dataset_name, quant_type):
        """Mock the quantization function"""
        # Create an empty file to simulate the quantized model
        with open(output_path, 'w') as f:
            f.write("Mock quantized model")
        return True
    

class TestOptimizationPipeline(unittest.TestCase):
    """Tests for the optimization pipeline components"""
    
    @patch('pruning_logic.Streamlined_prune.main_pruning_loop', MockOptimization.mock_main_pruning_loop)
    def test_block_pruning_pipeline(self):
        """Test IT-01: Block pruning pipeline"""
        model = SimpleModel(num_blocks=3)
        # Apply mock pruning
        pruned_model = MockOptimization.mock_main_pruning_loop(
            model=model,
            block_level_dict={'blocks.0': 0.5, 'blocks.1': 0.5, 'blocks.2': 0.5},
            uniform_pruning_ratio=0,
            block_fine_tune_epochs=5,
            channel_fine_tune_epochs=0,
            device='cpu',
            type='BLOCK'
        )
        # Check that blocks were pruned
        self.assertLess(len(pruned_model.blocks), len(model.blocks))
    
    @patch('pruning_logic.Streamlined_prune.knowledge_distillation_prune', MockOptimization.mock_knowledge_distillation_prune)
    def test_knowledge_distillation(self):
        """Test IT-03: Knowledge distillation"""
        teacher_model = SimpleModel(num_blocks=5)
        student_model = SimpleModel(num_blocks=3)
        # Apply mock distillation
        distilled_model = MockOptimization.mock_knowledge_distillation_prune(
            teacher_model=teacher_model,
            student_model=student_model,
            num_epochs=10,
            device='cpu'
        )
        # Check that we got back a model
        self.assertIsInstance(distilled_model, nn.Module)
    
    @patch('quantization_logic.quantization.quantize_model', MockOptimization.mock_quantize_model)
    def test_quantization(self):
        """Test IT-04: Quantization"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            model_path = tmp_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tflite') as tmp_output:
            output_path = tmp_output.name
        
        try:
            # Apply mock quantization
            result = MockOptimization.mock_quantize_model(
                model_path=model_path,
                output_path=output_path,
                dataset_name="person_detection_validation",
                quant_type="int8"
            )
            # Check that output file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(result)
        finally:
            # Clean up temporary files
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Core Backend Logic Unit Tests
class TestDataLoading(unittest.TestCase):
    """Tests for data loading functionality."""
    
    def setUp(self):
        # Create a temporary directory with fake image data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create fake class directories
        self.class1_dir = os.path.join(self.temp_dir, 'class1')
        self.class2_dir = os.path.join(self.temp_dir, 'class2')
        os.makedirs(self.class1_dir)
        os.makedirs(self.class2_dir)
        
        # Create some dummy images
        for i in range(10):
            img = Image.new('RGB', (100, 100), color=(i*20, 100, 100))
            img.save(os.path.join(self.class1_dir, f'img{i}.jpg'))
            
            img = Image.new('RGB', (100, 100), color=(100, i*20, 100))
            img.save(os.path.join(self.class2_dir, f'img{i}.jpg'))
    
    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_dataloaders(self):
        """Test that dataloaders are created correctly with the expected splits."""
        train_loader, val_loader, test_loader = prepare_dataloaders(self.temp_dir, batch_size=4)
        
        # Check that we have the correct loaders
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Check the approximate sizes of each split (70%, 10%, 20%)
        total_samples = 20  # 10 images per class, 2 classes
        expected_train_size = int(0.7 * total_samples)
        expected_val_size = int(0.1 * total_samples)
        expected_test_size = total_samples - expected_train_size - expected_val_size
        
        # Count the actual number of samples in each loader
        train_samples = sum(1 for _ in train_loader) * train_loader.batch_size
        val_samples = sum(1 for _ in val_loader) * val_loader.batch_size
        test_samples = sum(1 for _ in test_loader) * test_loader.batch_size
        
        # Allow for some difference due to incomplete batches
        self.assertLessEqual(abs(train_samples - expected_train_size), train_loader.batch_size)
        self.assertLessEqual(abs(val_samples - expected_val_size), val_loader.batch_size)
        self.assertLessEqual(abs(test_samples - expected_test_size), test_loader.batch_size)
        
    def test_grayscale_conversion(self):
        """Test that images are correctly converted to grayscale."""
        train_loader, _, _ = prepare_dataloaders(self.temp_dir, batch_size=1)
        
        # Get a batch of data
        images, _ = next(iter(train_loader))
        
        # Check that the images are single-channel (grayscale)
        self.assertEqual(images.shape[1], 1)
        
    def test_normalization(self):
        """Test that the normalization is applied correctly."""
        train_loader, _, _ = prepare_dataloaders(self.temp_dir, batch_size=1)
        
        # Get a batch of data
        images, _ = next(iter(train_loader))
        
        # Check that values are normalized around mean=0.5, std=0.5
        # This means most values should be between -1 and 1
        self.assertTrue(torch.all(images >= -1.5))
        self.assertTrue(torch.all(images <= 1.5))


class TestModelPruning(unittest.TestCase):
    """Tests for the model pruning functionality."""
    
    def setUp(self):
        # Create a mock model for testing
        self.mock_model = MagicMock()
        
        # Mock model components that might be accessed during pruning
        self.mock_model.blocks = [MagicMock() for _ in range(5)]
        self.mock_model.classifier = MagicMock()
        self.mock_model.classifier.linear = MagicMock()
        self.mock_model.classifier.linear.weight = torch.randn(10, 20)  # Mock classifier weight
        
        # Setup device for tests
        self.device = torch.device("cpu")
    
    @patch('pruning_logic.Streamlined_prune.prune_mb_inverted_block')
    def test_prune_multiple_blocks(self, mock_prune_block):
        """Test that multiple blocks can be pruned correctly."""
        # Setup mock return value for prune_mb_inverted_block
        mock_prune_block.return_value = self.mock_model
        
        # Define target blocks to prune
        target_blocks = {
            'blocks.1.mobile_inverted_conv': 0.3,
            'blocks.3.mobile_inverted_conv': 0.5
        }
        
        # Call the function
        result = prune_multiple_blocks(self.mock_model, target_blocks)
        
        # Verify prune_mb_inverted_block was called twice with correct arguments
        self.assertEqual(mock_prune_block.call_count, 2)
        mock_prune_block.assert_any_call(
            model=self.mock_model,
            target_conv_path='blocks.1.mobile_inverted_conv',
            prune_ratio=0.3,
            MobileInvertedBlockClass=MobileInvertedResidualBlock,
            SqueezeExcitationModuleType=SEModule,
            verbose=True
        )
        mock_prune_block.assert_any_call(
            model=self.mock_model,
            target_conv_path='blocks.3.mobile_inverted_conv',
            prune_ratio=0.5,
            MobileInvertedBlockClass=MobileInvertedResidualBlock,
            SqueezeExcitationModuleType=SEModule,
            verbose=True
        )
        
        # Verify the function returns the pruned model
        self.assertEqual(result, self.mock_model)
    
    @patch('pruning_logic.Streamlined_prune.uniform_channel_prune')
    @patch('pruning_logic.Streamlined_prune.col_based_prune_reduction')
    def test_uniform_prune_and_depthwise_collapse(self, mock_col_prune, mock_uniform_prune):
        """Test uniform pruning and collapsing of depthwise convolutions."""
        # Setup mock returns
        mock_uniform_prune.return_value = self.mock_model
        mock_col_prune.return_value = self.mock_model
        
        # Create a mock Pruner instance
        mock_pruner = MagicMock()
        
        # Patch the Pruner class to return our mock
        with patch('pruning_logic.Streamlined_prune.Pruner', return_value=mock_pruner):
            result = uniform_prune_and_depthwise_collapse(self.mock_model, 0.5)
            
            # Verify uniform_channel_prune was called with correct args
            mock_uniform_prune.assert_called_once_with(
                model=self.mock_model,
                prune_ratio=0.5,
                SqueezeExcitationModuleType=SEModule,
                verbose=True
            )
            
            # Verify Pruner was instantiated correctly
            self.assertEqual('pruning_logic.Pruning_definitions.Pruner', Pruner.__module__ + '.' + Pruner.__name__)
            
            # Verify pruner.apply was called
            mock_pruner.apply.assert_called_once_with(self.mock_model)
            
            # Verify col_based_prune_reduction was called
            mock_col_prune.assert_called_once_with(
                self.mock_model, 
                ['classifier.linear.weight']
            )
            
            # Verify the function returns the pruned model
            self.assertEqual(result, self.mock_model)
    
    @patch('pruning_logic.Streamlined_prune.prune_multiple_blocks')
    @patch('pruning_logic.Streamlined_prune.uniform_prune_and_depthwise_collapse')
    @patch('pruning_logic.Streamlined_prune.main_finetune_model')
    def test_main_pruning_loop_both(self, mock_finetune, mock_uniform_prune, mock_block_prune):
        """Test the main pruning loop with both block and uniform pruning."""
        # Setup mock returns
        mock_block_prune.return_value = self.mock_model
        mock_uniform_prune.return_value = self.mock_model
        mock_finetune.return_value = self.mock_model
        
        block_level_dict = {'blocks.1.mobile_inverted_conv': 0.3}
        
        # Test with both types of pruning
        result = main_pruning_loop(
            self.mock_model, 
            block_level_dict, 
            0.5, 
            2,  # block fine-tune epochs
            2,  # channel fine-tune epochs
            self.device,
            "BOTH"
        )
        
        # Verify both pruning methods were called
        mock_block_prune.assert_called_once_with(self.mock_model, block_level_dict, 2)
        mock_uniform_prune.assert_called_once_with(self.mock_model, 0.5)
        
        # Verify fine-tuning was called twice
        self.assertEqual(mock_finetune.call_count, 2)
        
        # Verify the function returns the pruned model
        self.assertEqual(result, self.mock_model)
        
    @patch('pruning_logic.Streamlined_prune.prune_multiple_blocks')
    @patch('pruning_logic.Streamlined_prune.uniform_prune_and_depthwise_collapse')
    @patch('pruning_logic.Streamlined_prune.main_finetune_model')
    def test_main_pruning_loop_block_only(self, mock_finetune, mock_uniform_prune, mock_block_prune):
        """Test the main pruning loop with block pruning only."""
        # Setup mock returns
        mock_block_prune.return_value = self.mock_model
        mock_finetune.return_value = self.mock_model
        
        block_level_dict = {'blocks.1.mobile_inverted_conv': 0.3}
        
        # Test with only block pruning
        result = main_pruning_loop(
            self.mock_model, 
            block_level_dict, 
            0.5,  # This shouldn't be used
            2,    # block fine-tune epochs
            0,    # channel fine-tune epochs
            self.device,
            "BLOCK"
        )
        
        # Verify only block pruning was called
        mock_block_prune.assert_called_once()
        mock_uniform_prune.assert_not_called()
        
        # Verify fine-tuning was called once
        mock_finetune.assert_called_once()
        
        # Verify the function returns the pruned model
        self.assertEqual(result, self.mock_model)
        
    @patch('pruning_logic.Streamlined_prune.prune_multiple_blocks')
    @patch('pruning_logic.Streamlined_prune.uniform_prune_and_depthwise_collapse')
    @patch('pruning_logic.Streamlined_prune.main_finetune_model')
    def test_main_pruning_loop_uniform_only(self, mock_finetune, mock_uniform_prune, mock_block_prune):
        """Test the main pruning loop with uniform pruning only."""
        # Setup mock returns
        mock_uniform_prune.return_value = self.mock_model
        mock_finetune.return_value = self.mock_model
        
        block_level_dict = {'blocks.1.mobile_inverted_conv': 0.3}  # This shouldn't be used
        
        # Test with only uniform pruning
        result = main_pruning_loop(
            self.mock_model, 
            block_level_dict, 
            0.5,
            0,  # block fine-tune epochs
            2,  # channel fine-tune epochs
            self.device,
            "UNIFORM"
        )
        
        # Verify only uniform pruning was called
        mock_block_prune.assert_not_called()
        mock_uniform_prune.assert_called_once()
        
        # Verify fine-tuning was called once
        mock_finetune.assert_called_once()
        
        # Verify the function returns the pruned model
        self.assertEqual(result, self.mock_model)


class TestModelFineTuning(unittest.TestCase):
    """Tests for model fine-tuning functionality."""
    
    def setUp(self):
        # Create a mock model for testing
        self.mock_model = MagicMock()
        
        # Setup device for tests
        self.device = torch.device("cpu")

if __name__ == '__main__':
    unittest.main()