import unittest
import torch
import copy
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys
import tempfile
from PIL import Image
import shutil

# Add the parent directory to sys.path to import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module under test - adjust the import path as needed
from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.tinynas.nn.networks.mcunets import MobileInvertedResidualBlock
from mcunet.utils.pytorch_modules import SEModule

# Import the functions to test
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
    

class TestEndToEndPruning(unittest.TestCase):
    """Integration tests for the full pruning workflow."""
    
    def setUp(self):
        # Try to load a real model for end-to-end testing
        try:
            # This is just a placeholder. You'll need to replace with actual model initialization
            self.model = MagicMock()
            self.model.to = MagicMock(return_value=self.model)
            self.device = torch.device("cpu")
        except Exception as e:
            self.skipTest(f"Skipping end-to-end test because model couldn't be loaded: {e}")
    
    @patch('pruning_logic.Streamlined_prune.evaluate')
    def test_test_model(self, mock_evaluate):
        from pruning_logic.Streamlined_prune import test_model 

        """Test the test_model function."""
        # Setup mock test_loader to return a couple of batches
        mock_batch1 = (torch.randn(4, 1, 96, 96), torch.randint(0, 2, (4,)))
        mock_batch2 = (torch.randn(4, 1, 96, 96), torch.randint(0, 2, (4,)))
        
        with patch('pruning_logic.Streamlined_prune.test_loader', __iter__=lambda _: iter([mock_batch1, mock_batch2])):
            # Mock evaluate to return some results
            mock_evaluate.return_value = {'accuracy': 0.85, 'loss': 0.2}
            
            # Call test_model
            result = test_model(self.model, self.device)
            
            # Verify evaluate was called with the correct arguments
            mock_evaluate.assert_called_once()
            args = mock_evaluate.call_args[0]
            self.assertEqual(args[0], self.model)
            self.assertEqual(len(args[1]), 2)  # Two batches
            self.assertEqual(args[2], self.device)
            self.assertEqual(mock_evaluate.call_args[1]['verbose'], True)
            
            # Verify the function returns the evaluation results
            self.assertEqual(result, {'accuracy': 0.85, 'loss': 0.2})
    
if __name__ == '__main__':
    unittest.main()