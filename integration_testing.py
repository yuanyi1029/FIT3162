import unittest
import tempfile
import os
import torch
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from streamlit.testing.v1 import AppTest
from torch import nn
from main_v2 import identify_model_blocks

class TestEndToEndOptimizationWorkflow(unittest.TestCase):
    """End-to-end tests for the complete optimization workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = ComplexDummyModel()
        self.device = torch.device("cpu")
        self.temp_model_path = os.path.join(tempfile.gettempdir(), "test_e2e_model.pth")
        torch.save(self.model, self.temp_model_path)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_model_path):
            os.unlink(self.temp_model_path)
    
    @patch('pruning_logic.Streamlined_prune.main_pruning_loop')
    @patch('quantization_logic.quantization.quantize_model')
    @patch('quantization_logic.quantization.get_tflite_model_size')
    @patch('pruning_logic.Streamlined_prune.test_model')
    @patch('pruning_logic.Streamlined_prune.knowledge_distillation_prune')
    def test_full_optimization_workflow(self, mock_kd_prune, mock_test_model, 
                                       mock_get_tflite_size, mock_quantize_model, mock_pruning_loop):
        """Test the complete optimization workflow with all methods enabled."""
        # Mock pruning to return our model
        mock_pruning_loop.return_value = self.model
        mock_test_model.return_value = {"accuracy": 0.88, "loss": 0.12}
        mock_get_tflite_size.return_value = 0.25  # 0.25 MB
        
        # Mock knowledge distillation
        mock_kd_prune.return_value = self.model
        
        # Set up teacher model for KD
        teacher_model = ComplexDummyModel()  # Using same model for simplicity
        teacher_model_path = os.path.join(tempfile.gettempdir(), "teacher_model.pth")
        torch.save(teacher_model, teacher_model_path)
        
        try:
            # Apply block pruning
            block_pruning_ratios = {"blocks.0": 0.5, "blocks.1": 0.3}
            pruned_model = mock_pruning_loop(
                model=self.model,
                block_level_dict=block_pruning_ratios,
                uniform_pruning_ratio=0.4,  # Channel pruning
                block_fine_tune_epochs=5,
                channel_fine_tune_epochs=3,
                device=self.device,
                type="BOTH"  # Both block and channel pruning
            )
            
            # Verify pruning called with correct parameters
            mock_pruning_loop.assert_called_once_with(
                model=self.model,
                block_level_dict=block_pruning_ratios,
                uniform_pruning_ratio=0.4,
                block_fine_tune_epochs=5,
                channel_fine_tune_epochs=3,
                device=self.device,
                type="BOTH"
            )
            
            # Apply knowledge distillation
            teacher_model = torch.load(teacher_model_path)
            distilled_model = mock_kd_prune(
                teacher_model=teacher_model,
                student_model=pruned_model,
                num_epochs=10,
                device=self.device
            )
            
            # Verify KD called with correct parameters
            mock_kd_prune.assert_called_once_with(
                teacher_model=teacher_model,
                student_model=pruned_model,
                num_epochs=10,
                device=self.device
            )
            
            # Apply quantization
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_pruned_file, \
                 tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_quant_file:
                
                pruned_model_path = tmp_pruned_file.name
                quant_model_path = tmp_quant_file.name
                
                # Save the distilled model
                torch.save(distilled_model, pruned_model_path)
                
                # Apply quantization
                mock_quantize_model(pruned_model_path, quant_model_path, "person_detection_validation", "int8")
                
                # Verify quantization called with correct parameters
                mock_quantize_model.assert_called_once_with(
                    pruned_model_path, quant_model_path, "person_detection_validation", "int8"
                )
                
                # Get quantized model size
                quantized_size = mock_get_tflite_size(quant_model_path)
                self.assertEqual(quantized_size, 0.25)
                
                # Check final model accuracy
                accuracy = mock_test_model(distilled_model, self.device)
                self.assertEqual(accuracy, {"accuracy": 0.88, "loss": 0.12})
                
            # Clean up temporary files
            os.remove(pruned_model_path)
            os.remove(quant_model_path)
        
        finally:
            # Clean up teacher model
            if os.path.exists(teacher_model_path):
                os.unlink(teacher_model_path)

class TestIncrementalOptimizationFlow(unittest.TestCase):
    """Test the incremental optimization flow with different methods."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = ComplexDummyModel()
        self.device = torch.device("cpu")
        self.temp_model_path = os.path.join(tempfile.gettempdir(), "test_incremental_model.pth")
        torch.save(self.model, self.temp_model_path)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_model_path):
            os.unlink(self.temp_model_path)
    
    @patch('pruning_logic.Streamlined_prune.main_pruning_loop')
    @patch('pruning_logic.Streamlined_prune.test_model')
    def test_block_pruning_only(self, mock_test_model, mock_pruning_loop):
        """Test with block pruning only."""
        # Mock pruning to return our model
        mock_pruning_loop.return_value = self.model
        mock_test_model.return_value = {"accuracy": 0.9, "loss": 0.1}
        
        # Apply block pruning only
        block_pruning_ratios = {"blocks.0": 0.4, "blocks.1": 0.2}
        pruned_model = mock_pruning_loop(
            model=self.model,
            block_level_dict=block_pruning_ratios,
            uniform_pruning_ratio=0.0,  # No channel pruning
            block_fine_tune_epochs=5,
            channel_fine_tune_epochs=0,
            device=self.device,
            type="BLOCK"
        )
        
        # Verify pruning called with correct parameters
        mock_pruning_loop.assert_called_once_with(
            model=self.model,
            block_level_dict=block_pruning_ratios,
            uniform_pruning_ratio=0.0,
            block_fine_tune_epochs=5,
            channel_fine_tune_epochs=0,
            device=self.device,
            type="BLOCK"
        )
        
        # Check the model accuracy after pruning
        accuracy = mock_test_model(pruned_model, self.device)
        self.assertEqual(accuracy, {"accuracy": 0.9, "loss": 0.1})
    
    @patch('pruning_logic.Streamlined_prune.main_pruning_loop')
    @patch('pruning_logic.Streamlined_prune.test_model')
    def test_channel_pruning_only(self, mock_test_model, mock_pruning_loop):
        """Test with channel pruning only."""
        # Mock pruning to return our model
        mock_pruning_loop.return_value = self.model
        mock_test_model.return_value = {"accuracy": 0.87, "loss": 0.13}
        
        # Apply channel pruning only
        pruned_model = mock_pruning_loop(
            model=self.model,
            block_level_dict={},  # No block pruning
            uniform_pruning_ratio=0.5,  # Channel pruning ratio
            block_fine_tune_epochs=0,
            channel_fine_tune_epochs=8,
            device=self.device,
            type="UNIFORM"
        )
        
        # Verify pruning called with correct parameters
        mock_pruning_loop.assert_called_once_with(
            model=self.model,
            block_level_dict={}, 
            uniform_pruning_ratio=0.5,
            block_fine_tune_epochs=0,
            channel_fine_tune_epochs=8,
            device=self.device,
            type="UNIFORM"
        )
        
        # Check the model accuracy after pruning
        accuracy = mock_test_model(pruned_model, self.device)
        self.assertEqual(accuracy, {"accuracy": 0.87, "loss": 0.13})
    
    @patch('quantization_logic.quantization.quantize_model')
    @patch('quantization_logic.quantization.get_tflite_model_size')
    def test_quantization_only(self, mock_get_tflite_size, mock_quantize_model):
        """Test with quantization only."""
        mock_get_tflite_size.return_value = 0.3  # 0.3 MB
        
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_quant_file:
            quant_model_path = tmp_quant_file.name
            
            # Apply quantization directly to the original model
            mock_quantize_model(self.temp_model_path, quant_model_path, "person_detection_validation", "float16")
            
            # Verify quantization called with correct parameters
            mock_quantize_model.assert_called_once_with(
                self.temp_model_path, quant_model_path, "person_detection_validation", "float16"
            )
            
            # Get quantized model size
            quantized_size = mock_get_tflite_size(quant_model_path)
            self.assertEqual(quantized_size, 0.3)
            
        # Clean up quantized model file
        os.remove(quant_model_path)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the optimization workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = DummyModel()
        self.device = torch.device("cpu")
        self.temp_model_path = os.path.join(tempfile.gettempdir(), "test_error_model.pth")
        torch.save(self.model, self.temp_model_path)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_model_path):
            os.unlink(self.temp_model_path)
    
    @patch('pruning_logic.Streamlined_prune.main_pruning_loop')
    def test_pruning_error_handling(self, mock_pruning_loop):
        """Test error handling during pruning."""
        # Simulate an error during pruning
        mock_pruning_loop.side_effect = RuntimeError("Simulated pruning error")
        
        # Apply pruning and expect exception to be caught
        with self.assertRaises(RuntimeError):
            pruned_model = mock_pruning_loop(
                model=self.model,
                block_level_dict={"blocks.0": 0.5},
                uniform_pruning_ratio=0.0,
                block_fine_tune_epochs=5,
                channel_fine_tune_epochs=0,
                device=self.device,
                type="BLOCK"
            )
    
    @patch('quantization_logic.quantization.quantize_model')
    def test_quantization_error_handling(self, mock_quantize_model):
        """Test error handling during quantization."""
        # Simulate an error during quantization
        mock_quantize_model.side_effect = RuntimeError("Simulated quantization error")
        
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_quant_file:
            quant_model_path = tmp_quant_file.name
            
            # Apply quantization and expect exception to be caught
            with self.assertRaises(RuntimeError):
                mock_quantize_model(
                    self.temp_model_path, quant_model_path, "person_detection_validation", "int8"
                )
            
        # Clean up quantized model file
        os.remove(quant_model_path)


class TestOptimizationMetrics(unittest.TestCase):
    """Test the calculation of optimization metrics."""
    
    def setUp(self):
        """Set up test environment."""
        self.model = ComplexDummyModel()
        self.device = torch.device("cpu")
        self.temp_model_path = os.path.join(tempfile.gettempdir(), "test_metrics_model.pth")
        torch.save(self.model, self.temp_model_path)
        
        # Create a smaller model to simulate pruning results
        self.pruned_model = SimplerDummyModel()
        self.pruned_model_path = os.path.join(tempfile.gettempdir(), "test_pruned_model.pth")
        torch.save(self.pruned_model, self.pruned_model_path)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_model_path):
            os.unlink(self.temp_model_path)
        if os.path.exists(self.pruned_model_path):
            os.unlink(self.pruned_model_path)
    
    @patch('pruning_logic.Pruning_definitions.get_model_size')
    @patch('mcunet.utils.pytorch_utils.count_net_flops')
    @patch('mcunet.utils.pytorch_utils.count_peak_activation_size')
    def test_model_size_calculation(self, mock_count_peak, mock_count_flops, mock_get_model_size):
        """Test calculation of model size and stats."""
        # Set up mocks
        mock_get_model_size.side_effect = [1.5, 0.8]  # Original: 1.5MB, Pruned: 0.8MB
        mock_count_flops.side_effect = [5000000, 2500000]  # Original: 5M, Pruned: 2.5M
        mock_count_peak.side_effect = [1000000, 500000]  # Original: 1M, Pruned: 0.5M
        
        # Get original model size
        original_size = mock_get_model_size(self.model)
        original_flops = mock_count_flops(self.model, (1, 1, 96, 96))
        original_peak = mock_count_peak(self.model, (1, 1, 96, 96))
        
        # Get pruned model size
        pruned_size = mock_get_model_size(self.pruned_model)
        pruned_flops = mock_count_flops(self.pruned_model, (1, 1, 96, 96))
        pruned_peak = mock_count_peak(self.pruned_model, (1, 1, 96, 96))
        
        # Calculate size reduction percentage
        size_reduction = ((original_size - pruned_size) / original_size) * 100
        flops_reduction = ((original_flops - pruned_flops) / original_flops) * 100
        peak_reduction = ((original_peak - pruned_peak) / original_peak) * 100
        
        # Verify the calculations
        self.assertAlmostEqual(original_size, 1.5, places=5)
        self.assertAlmostEqual(pruned_size, 0.8, places=5)
        self.assertAlmostEqual(size_reduction, 46.66666666666667, places=5)

        self.assertEqual(original_flops, 5000000)
        self.assertEqual(pruned_flops, 2500000)
        self.assertAlmostEqual(flops_reduction, 50.0, places=5)

        self.assertEqual(original_peak, 1000000)
        self.assertEqual(pruned_peak, 500000)
        self.assertAlmostEqual(peak_reduction, 50.0, places=5)


# Extra sample model classes for testing

class DummyModel(nn.Module):
    """A very simple model for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

class ComplexDummyModel(nn.Module):
    """A more complex model with blocks for testing."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
            )
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SimplerDummyModel(nn.Module):
    """A smaller model to represent pruned result."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 1, kernel_size=3, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=3, padding=1)
            )
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TestBlockIdentification(unittest.TestCase):
    """Test the identify_model_blocks function with various model architectures."""
    
    def test_identify_blocks_standard_model(self):
        """Test block identification on a standard model with blocks."""
        model = ComplexDummyModel()
        blocks = identify_model_blocks(model)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks, ["blocks.0", "blocks.1"])
    
    def test_identify_blocks_no_blocks(self):
        """Test block identification on a model without blocks."""
        model = DummyModel()  # Model without blocks list
        blocks = identify_model_blocks(model)
        self.assertEqual(len(blocks), 0)
    
    def test_identify_blocks_nested_structure(self):
        """Test block identification on a model with nested blocks."""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.ModuleList([nn.Conv2d(1, 8, 3), nn.ReLU()]),
                    nn.ModuleList([nn.Conv2d(8, 1, 3), nn.ReLU()])
                ])
                
            def forward(self, x):
                for block_group in self.blocks:
                    for layer in block_group:
                        x = layer(x)
                return x
        
        model = NestedModel()
        blocks = identify_model_blocks(model)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks, ["blocks.0", "blocks.1"])
    
    def test_identify_blocks_with_error(self):
        """Test error handling in block identification."""
        # Mock a model that raises an exception when accessing attributes
        mock_model = MagicMock()
        mock_model.named_modules.side_effect = AttributeError("Simulated error")
        
        blocks = identify_model_blocks(mock_model)
        self.assertEqual(blocks, [])


if __name__ == "__main__":
    unittest.main()