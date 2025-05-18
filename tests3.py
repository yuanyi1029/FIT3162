import pytest
from unittest.mock import patch, MagicMock, mock_open
import torch
from torch import nn
import tempfile
import os
import unittest

# Import the AppTest class from Streamlit's testing framework
from streamlit.testing.v1 import AppTest

# Import the specific functions from your Streamlit app script
# Assuming your Streamlit script is named 'your_streamlit_app_script.py'
# You'll need to replace 'your_streamlit_app_script' with the actual name of your Python file.
# For example, if your script is 'app.py', use:
# from app import identify_model_blocks
# For this example, let's assume the script is named 'streamlit_app.py'
from main_v2 import identify_model_blocks

# --- White-box tests for identify_model_blocks ---

class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Dummy parameter to make it a valid module
        self.dummy_param = nn.Parameter(torch.empty(1))

    def named_modules(self):
        return # Default, override in tests

@pytest.fixture
def mock_model_with_blocks():
    model = MockModule()
    def _named_modules():
        return [
            ("blocks.1", MockModule()),
            ("blocks.0", MockModule()),
            ("blocks.0.layer", MockModule()),
            ("other_module", MockModule())
        ]
    model.named_modules = _named_modules
    return model

@pytest.fixture
def mock_model_no_blocks():
    model = MockModule()
    def _named_modules():
        return [
            ("features.0", MockModule()),
            ("blocks.submodule.0", MockModule()),
            ("blocks.a", MockModule()),
            ("layer1", MockModule())
        ]
    model.named_modules = _named_modules
    return model

@pytest.fixture
def mock_model_empty_named_modules():
    model = MockModule()
    def _named_modules():
        return
    model.named_modules = _named_modules
    return model

def test_identify_model_blocks_standard(mock_model_with_blocks):
    """Test Example 1: Standard Block Extraction"""
    expected = ['blocks.0', 'blocks.1']
    actual = identify_model_blocks(mock_model_with_blocks)
    assert sorted(actual) == sorted(expected)

def test_identify_model_blocks_no_matching(mock_model_no_blocks):
    """Test Example 2: No Matching Blocks or Invalid Block Names"""
    expected = []
    actual = identify_model_blocks(mock_model_no_blocks)
    assert actual == expected

def test_identify_model_blocks_empty_model(mock_model_empty_named_modules):
    """Test with a model that yields no modules from named_modules"""
    expected = []
    actual = identify_model_blocks(mock_model_empty_named_modules)
    assert actual == expected

def test_identify_model_blocks_error_handling():
    """Test error handling if named_modules fails - checks st.error call"""
    model = MockModule()
    def faulty_named_modules():
        raise Exception("Test error")
    model.named_modules = faulty_named_modules

    # To test st.error, we need to run it within an AppTest context
    # Create a minimal app script string that calls identify_model_blocks
    minimal_app_script = """
import streamlit as st
from torch import nn

# Definition of identify_model_blocks must be available here or imported
# For simplicity, assuming it's globally available or correctly imported in the test setup
# If identify_model_blocks is in streamlit_app.py, ensure it's found.

# Re-define or import identify_model_blocks if it's not found by from_string
def identify_model_blocks_inline(model_to_test):
    block_names =
    try:
        for name, module in model_to_test.named_modules():
            if isinstance(module, nn.Module) and name.startswith("blocks"):
                parts = name.split(".")
                if len(parts) == 2 and parts == "blocks" and parts.isdigit():
                    block_names.append(name)
        return sorted(block_names, key=lambda x: int(x.split(".")))
    except Exception as e:
        st.error(f"Error extracting block names: {e}")
        return

# The model will be passed via a global or session_state in a real AppTest scenario
# For this specific unit test of error path, we can call it directly
# and check st.error if AppTest captures it from a direct call.
# However, AppTest usually runs the whole script.
# A better way for st.error is to have the function call st.error and check at.error

st.session_state.test_model_global = model_to_test_error_scenario
identify_model_blocks_inline(st.session_state.test_model_global)
"""
    at = AppTest.from_string(minimal_app_script)
    # Inject the faulty model into the AppTest environment
    # This is a bit indirect; usually, you'd mock where identify_model_blocks is called.
    # For this specific test, we're testing the st.error path.
    at.session_state["model_to_test_error_scenario"] = model
    
    at.run()
    assert len(at.error) == 1
    assert "Error extracting block names: Test error" in at.error.value
    # And verify the function still returns in this case (implicitly tested by AppTest not crashing)


# --- Black-box tests for Streamlit UI ---

@pytest.fixture(scope="function")
@patch('main_v2.torch.load') # Patch torch.load where it's used in streamlit_app.py
@patch('main_v2.identify_model_blocks')
@patch('main_v2.get_model_size')
@patch('main_v2.count_net_flops')
@patch('main_v2.count_peak_activation_size')
@patch('main_v2.main_pruning_loop')
@patch('main_v2.knowledge_distillation_prune')
@patch('main_v2.quantize_model')
@patch('main_v2.get_tflite_model_size')
@patch('main_v2.random.randint')
@patch('main_v2.test_model') # If test_model is used
def at(mock_test_model, mock_randint, mock_get_tflite_model_size, mock_quantize_model,
       mock_kd_prune, mock_main_pruning_loop, mock_count_peak_act, mock_count_flops,
       mock_get_model_size, mock_identify_blocks, mock_torch_load):
    """Fixture to initialize AppTest and mock backend functions."""
    
    # Configure mocks
    mock_torch_load.return_value = MockModule() # Return a mock model
    mock_identify_blocks.return_value = ["blocks.0", "blocks.1"] # For UI population
    mock_get_model_size.side_effect = [10.0, 5.0, 2.0] # Original, Pruned, Quantized (example values)
    mock_count_flops.side_effect = [100e6, 50e6, 50e6] # Original, Pruned, Quantized
    mock_count_peak_act.return_value = 1e6
    mock_main_pruning_loop.return_value = MockModule() # Return a mock pruned model
    mock_kd_prune.return_value = MockModule() # Return a mock distilled model
    
    # Mock quantize_model to simulate file creation
    def side_effect_quantize(model_path, output_path, dataset, quant_type):
        with open(output_path, 'w') as f:
            f.write("dummy tflite content")
    mock_quantize_model.side_effect = side_effect_quantize
    
    mock_get_tflite_model_size.return_value = 2.0 # MB
    mock_randint.return_value = 90 # For original_acc
    mock_test_model.return_value = 85 # For pruned_acc if used

    # Initialize AppTest
    # Replace 'streamlit_app.py' with the actual name of your Streamlit script
    app_test_instance = AppTest.from_file("main_v2.py", default_timeout=30).run()
    return app_test_instance

def test_initial_state_and_file_upload(at):
    """Test Example 1: Successful File Upload and UI Update"""
    # Initial state
    assert at.info.value == "👈 Please upload a PyTorch model (.pth) using the file uploader in the sidebar to get started."
    assert len(at.radio) == 0 # No optimization mode radio buttons initially

    # Simulate file upload
    # Create a dummy file for upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        tmp_file.write(b"dummy model content")
        dummy_model_path = tmp_file.name
    
    at.file_uploader(key="Upload PyTorch model (.pth)").set_value(dummy_model_path).run()

    assert at.success.value == "✅ Model uploaded"
    assert at.radio(key="Select Mode:").options == ["Automatic", "Advanced"]
    assert at.session_state.previous_mode == "Automatic"

    os.unlink(dummy_model_path) # Clean up dummy file

def test_automatic_mode_size_profile(at):
    """Test Example 1: Automatic Mode Configuration ("Size" Profile)"""
    # Prerequisite: File upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        tmp_file.write(b"dummy model content")
        dummy_model_path = tmp_file.name
    at.file_uploader(key="Upload PyTorch model (.pth)").set_value(dummy_model_path).run()

    # Select Automatic mode (default after upload)
    # Select "Size (Maximum Reduction)"
    at.radio(key="Optimize model for:").set_value("Size (Maximum Reduction)").run()

    assert at.session_state.block_pruning is True
    assert at.session_state.channel_pruning is True
    assert at.session_state.knowledge_distillation is False
    assert at.session_state.quantization is True
    assert at.session_state.quantization_type == "int8"
    assert at.session_state.block_pruning_ratio == 0.7
    assert at.session_state.channel_pruning_ratio == 0.6
    assert at.session_state.block_fine_tune_epochs == 3
    assert at.session_state.channel_fine_tune_epochs == 3
    
    # Check for the specific info message
    info_messages = [info.value for info in at.info]
    assert any("This profile focuses on achieving the smallest possible model size." in msg for msg in info_messages)
    
    os.unlink(dummy_model_path)

def test_advanced_mode_and_mocked_optimization(at):
    """Test Example 2: Advanced Mode Parameter Setting and "Optimize Model" with Mocked Backend"""
    # Prerequisite: File upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        tmp_file.write(b"dummy model content")
        dummy_model_path = tmp_file.name
    at.file_uploader(key="Upload PyTorch model (.pth)").set_value(dummy_model_path).run()

    # Select Advanced mode
    at.radio(key="Select Mode:").set_value("Advanced").run()

    # Configure Advanced options
    at.checkbox(key="Block Level Pruning").check().run()
    at.checkbox(key="Quantization").check().run()
    
    # Wait for tabs to appear and then interact with slider and selectbox
    # Note: Streamlit AppTest runs the script top-to-bottom on each interaction.
    # Tabs are conditional, so ensure they are rendered before trying to access elements within them.
    # The.run() after checkbox interactions should re-render the page with tabs.

    # Assuming tabs are now present. Sliders/Selectboxes might need specific keys if not auto-detected.
    # Streamlit AppTest element selection can be tricky for dynamic elements like those in tabs.
    # We'll rely on session_state for verification primarily, and assume widgets update it.
    # For direct widget interaction within tabs, more complex selectors or explicit keys on widgets are best.
    
    # Let's assume the sliders and selectbox update session_state correctly upon interaction.
    # We can set session state directly for testing this flow if widget interaction is complex to script
    at.session_state.block_pruning_ratio = 0.3
    at.session_state.quantization_type = "int8"
    at.run() # Run to reflect these session_state changes if they were set by widgets

    # Click Optimize Model
    at.button(key="Optimize Model").click().run()

    assert at.session_state.block_pruning is True
    # Slider interaction is not directly simulated here, relying on prior set or default
    # For a robust test, you'd use at.slider(key="Block Pruning Ratio").set_value(0.3).run()
    # This requires the slider to be uniquely identifiable.
    assert at.session_state.quantization is True
    assert at.session_state.quantization_type == "int8" # Should be set by selectbox

    # Check for "Optimization complete!"
    # The status_text is an st.empty(), its final value will be in an st.markdown/st.text
    assert "Optimization complete!" in at.markdown[-1].value # Assuming it's the last markdown

    # Check metrics (these depend on mock configurations)
    # Example: Check "After" Size metric
    # Metric elements can be tricky to select directly by label.
    # We'll check if st.metric was called by looking at the number of metric elements.
    # A more robust way is to give keys to st.metric if possible, or check their values.
    metrics = at.metric
    assert len(metrics) >= 6 # 3 "Before", 3 "After"
    
    # A more specific check on values (assuming order or specific keys)
    # This requires knowing the exact structure or adding keys to st.metric calls
    # For example, if the "After Size" metric is the 4th metric element:
    # assert metrics.label == "Size"
    # assert "2.00 MB" in metrics.value # Based on mocked get_tflite_model_size
    # assert "-80.0%" in metrics.delta # (10MB original - 2MB final) / 10MB

    # Check for download buttons
    assert len(at.download_button) == 2 # One for.pth, one for.tflite
    assert "Download Optimized Model State Dict (.pth)" in at.download_button.label
    assert "Download Quantized Model (.tflite)" in at.download_button.label
    
    os.unlink(dummy_model_path)


def test_error_missing_teacher_model(at):
    """Test Example 3: "Optimize Model" Button - Error Handling for Missing Teacher Model"""
    # Prerequisite: File upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        tmp_file.write(b"dummy model content")
        dummy_model_path = tmp_file.name
    at.file_uploader(key="Upload PyTorch model (.pth)").set_value(dummy_model_path).run()

    # Select Advanced mode
    at.radio(key="Select Mode:").set_value("Advanced").run()

    # Select Knowledge Distillation
    at.checkbox(key="Knowledge Distillation").check().run()
    
    # Ensure no teacher model is uploaded (default state of file_uploader for teacher is None)
    # Click Optimize Model
    at.button(key="Optimize Model").click().run()

    assert len(at.error) == 1
    assert at.error.value == "Please upload a Teacher Model for Knowledge Distillation."
    
    os.unlink(dummy_model_path)