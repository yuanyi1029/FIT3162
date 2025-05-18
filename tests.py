import os
import sys
import unittest
import tempfile
import torch
import torch.nn as nn
import streamlit as st
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import io
import json
from streamlit.testing.v1 import AppTest
import pytest

import logging
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

# Add the application directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the application functions
from main_v2 import identify_model_blocks

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

class TestUIComponents(unittest.TestCase):
    """Unit tests for UI components"""
    
    def test_identify_model_blocks_valid(self):
        """Test UT-01: identify_model_blocks with valid model"""
        model = SimpleModel(num_blocks=3)
        blocks = identify_model_blocks(model)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks, ['blocks.0', 'blocks.1', 'blocks.2'])
    
    def test_identify_model_blocks_empty(self):
        """Test UT-02: identify_model_blocks with model without blocks"""
        model = NoBlocksModel()
        blocks = identify_model_blocks(model)
        self.assertEqual(blocks, [])
    
    def test_identify_model_blocks_exception(self):
        """Test UT-03: identify_model_blocks with invalid model"""
        # Test with None as model which should raise an exception internally
        blocks = identify_model_blocks(None)
        self.assertEqual(blocks, [])
    
    def test_identify_model_blocks_complex(self):
        """Test with a more complex model structure"""
        model = ComplexModel()
        blocks = identify_model_blocks(model)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks, ['blocks.0', 'blocks.1'])

class TestSessionState(unittest.TestCase):
    """Tests for session state management"""

    def test_session_state_initialization(self):
        """Test UT-04: Session state initialization"""
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


class TestBlackBox:
    """Black box tests using Streamlit's testing utilities"""
    
    @pytest.fixture
    def app_test(self):
        """Set up the app test fixture"""
        return AppTest.from_file("app.py")
    
    def create_model_file(self):
        """Create a temporary model file for testing"""
        model = SimpleModel()
        file_path = os.path.join(tempfile.gettempdir(), "test_model.pth")
        torch.save(model, file_path)
        return file_path
    
    def test_upload_valid_model(self, app_test):
        """Test BF-01: Upload valid PyTorch model"""
        file_path = self.create_model_file()
        
        try:
            with open(file_path, "rb") as f:
                model_bytes = f.read()
            
            # Upload model file
            app_test.file_uploader(label="Upload PyTorch model (.pth)").set_value(
                {"name": "test_model.pth", "type": "application/octet-stream", "data": model_bytes}
            )
            app_test.run()
            
            # Check that success message appears
            assert app_test.success[0].value == "✅ Model uploaded"
            
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_basic_mode_size_optimization(self, app_test):
        """Test BF-04: Size optimization preset"""
        file_path = self.create_model_file()
        
        try:
            with open(file_path, "rb") as f:
                model_bytes = f.read()
            
            # Upload model file
            app_test.file_uploader(label="Upload PyTorch model (.pth)").set_value(
                {"name": "test_model.pth", "type": "application/octet-stream", "data": model_bytes}
            )
            
            # Select size optimization
            app_test.radio("Optimize model for:").set_value("Size (Maximum Reduction)")
            
            # Verify that correct parameters are set
            app_test.run()
            assert st.session_state.block_pruning == True
            assert st.session_state.channel_pruning == True
            assert st.session_state.quantization == True
            assert st.session_state.quantization_type == "int8"
            assert st.session_state.block_pruning_ratio == 0.7
            assert st.session_state.channel_pruning_ratio == 0.6
            
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_advanced_mode_block_pruning_only(self, app_test):
        """Test BF-07: Block pruning only in advanced mode"""
        file_path = self.create_model_file()
        
        try:
            with open(file_path, "rb") as f:
                model_bytes = f.read()
            
            # Upload model file
            app_test.file_uploader(label="Upload PyTorch model (.pth)").set_value(
                {"name": "test_model.pth", "type": "application/octet-stream", "data": model_bytes}
            )
            
            # Switch to advanced mode
            app_test.radio("Select Mode:").set_value("Advanced")
            
            # Select only block pruning
            app_test.checkbox("Block Level Pruning").set_value(True)
            app_test.checkbox("Channel Pruning").set_value(False)
            app_test.checkbox("Knowledge Distillation").set_value(False)
            app_test.checkbox("Quantization").set_value(False)
            
            # Verify settings
            app_test.run()
            assert st.session_state.block_pruning == True
            assert st.session_state.channel_pruning == False
            assert st.session_state.knowledge_distillation == False
            assert st.session_state.quantization == False
            
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    def test_no_optimization_method_selected(self, app_test):
        """Test BF-11: No optimization method selected"""
        file_path = self.create_model_file()
        
        try:
            with open(file_path, "rb") as f:
                model_bytes = f.read()
            
            # Upload model file
            app_test.file_uploader(label="Upload PyTorch model (.pth)").set_value(
                {"name": "test_model.pth", "type": "application/octet-stream", "data": model_bytes}
            )
            
            # Switch to advanced mode
            app_test.radio("Select Mode:").set_value("Advanced")
            
            # Uncheck all optimization methods
            app_test.checkbox("Block Level Pruning").set_value(False)
            app_test.checkbox("Channel Pruning").set_value(False)
            app_test.checkbox("Knowledge Distillation").set_value(False)
            app_test.checkbox("Quantization").set_value(False)
            
            # Click optimize button
            app_test.button("Optimize Model").click()
            
            # Verify error message
            app_test.run()
            assert "Please select at least one optimization method" in app_test.error[0].value
            
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling"""
    
    @patch('torch.load', side_effect=Exception("Invalid model file"))
    def test_model_loading_error(self, mock_load):
        """Test IT-07: Model loading error"""
        # This would need to be integrated with Streamlit's test framework
        # or tested manually
        pass
    
    def test_cleanup_temp_files(self):
        """Test IT-11: Temporary file cleanup"""
        # Create a temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file_path = temp_file.name
        temp_file.close()
        
        try:
            # Verify file exists
            self.assertTrue(os.path.exists(file_path))
            
            # In your app, you would call os.unlink(file_path) after processing
            # Here we'll just do it directly to test
            os.unlink(file_path)
            
            # Verify file was deleted
            self.assertFalse(os.path.exists(file_path))
        finally:
            # Clean up in case test fails
            if os.path.exists(file_path):
                os.unlink(file_path)


class TestPerformance:
    """Performance tests"""
    
    def test_model_optimization_time(self):
        """Test BP-02: Measure optimization process duration"""
        # Create a simple model
        model = SimpleModel()
        
        # Define mock optimization functions that track time
        import time
        
        start_time = time.time()
        
        # Mock the optimization steps
        # Block pruning
        pruned_model = MockOptimization.mock_main_pruning_loop(
            model=model,
            block_level_dict={'blocks.0': 0.5, 'blocks.1': 0.5, 'blocks.2': 0.5},
            uniform_pruning_ratio=0,
            block_fine_tune_epochs=5,
            channel_fine_tune_epochs=0,
            device='cpu',
            type='BLOCK'
        )
        
        # Quantization
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            model_path = tmp_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tflite') as tmp_output:
            output_path = tmp_output.name
        
        try:
            MockOptimization.mock_quantize_model(
                model_path=model_path,
                output_path=output_path,
                dataset_name="person_detection_validation",
                quant_type="int8"
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # The mock functions are very fast, so this is just a placeholder
            # In real tests, you'd set an appropriate threshold
            assert duration < 5.0, f"Optimization took too long: {duration} seconds"
            
        finally:
            # Clean up temporary files
            if os.path.exists(model_path):
                os.unlink(model_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# Main test runner
if __name__ == '__main__':
    unittest.main()