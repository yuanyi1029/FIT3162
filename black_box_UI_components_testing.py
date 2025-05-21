import os
import tempfile
import torch
import unittest
import streamlit as st # Needed for patching
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
from test_models import SimpleModel, DummyModel, ComplexDummyModel, SimplerDummyModel


# Mock the UploadedFile class that st.file_uploader will return when patched
class MockUploadedFile:
    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self._data = data
        self._read_pos = 0

    def read(self, size=-1):
        if self._read_pos >= len(self._data): return b''
        end_pos = len(self._data) if size == -1 else self._read_pos + size
        data_chunk = self._data[self._read_pos:end_pos]
        self._read_pos += len(data_chunk)
        return data_chunk

    def getvalue(self):
        return self._data

    def seek(self, offset, whence=0):
        if whence == 0: self._read_pos = offset
        elif whence == 1: self._read_pos += offset
        elif whence == 2: self._read_pos = len(self._data) + offset
        else: raise ValueError("invalid whence")
        self._read_pos = max(0, min(self._read_pos, len(self._data)))
        return self._read_pos

    def tell(self):
        return self._read_pos

    def __len__(self):
        return len(self._data)


# Base class for common setup/teardown for UI tests
class BaseUITest(unittest.TestCase):
    def setUp(self):
        # Create a dummy .pth model file for the patched file_uploader
        self.model = SimpleModel(num_blocks=3) # Use num_blocks=3 as per SimpleModel's init in test_models.py
        self.model_file_path = os.path.join(tempfile.gettempdir(), f"test_model_{os.getpid()}_{self._testMethodName}.pth")
        torch.save(self.model.state_dict(), self.model_file_path)

        with open(self.model_file_path, "rb") as f:
            self.model_bytes = f.read()

        # Patch torch.load that main_v2.py calls to load the uploaded file.
        # Make it return our SimpleModel mock instance, which can then be used by identify_model_blocks.
        self.patcher_torch_load = patch('torch.load', return_value=self.model)
        self.mock_torch_load = self.patcher_torch_load.start()

        # Patch identify_model_blocks if it's imported in main_v2.py and needs specific behavior.
        # Ensure the mock return value aligns with SimpleModel structure from test_models.py
        # SimpleModel has self.blocks = nn.ModuleList([nn.Linear(10,10) for _ in range(num_blocks)])
        # so identify_model_blocks should return ['blocks.0', 'blocks.1', 'blocks.2'] for num_blocks=3
        self.patcher_identify_blocks = patch('main_v2.identify_model_blocks', return_value=[f'blocks.{i}' for i in range(3)])
        self.mock_identify_blocks = self.patcher_identify_blocks.start()

        # Patch st.file_uploader to return our MockUploadedFile
        self.patcher_file_uploader = patch('streamlit.file_uploader', return_value=MockUploadedFile(
            name="test_model.pth", type="application/octet-stream", data=self.model_bytes
        ))
        self.mock_file_uploader = self.patcher_file_uploader.start()


    def tearDown(self):
        if os.path.exists(self.model_file_path):
            os.unlink(self.model_file_path)
        self.patcher_torch_load.stop()
        self.patcher_identify_blocks.stop()
        self.patcher_file_uploader.stop() # Stop the patcher


# UI Interaction Test Classes
class TestBasicModeSizeOptimization(BaseUITest):
    def test_basic_mode_size_optimization(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Automatic' mode with 'Size (Maximum Reduction)' selected.")
        print("       Verifies that session state variables for pruning and quantization are set correctly.")
        print("Expected: session_state.block_pruning = True")
        print("          session_state.channel_pruning = True")
        print("          session_state.quantization = True")
        print("          session_state.quantization_type = 'float16'")
        print("          session_state.block_pruning_ratio = 0.5")
        print("          session_state.channel_pruning_ratio = 0.4")
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        
        # First run: App renders, st.file_uploader is called (patched to return mock file).
        # This run also causes the `if uploaded_file:` block to execute, revealing other widgets.
        app_test.run(timeout=30) 

        # Now, interact with the radio buttons located in the sidebar
        app_test.sidebar.radio("Select Mode:").set_value("Automatic")
        app_test.sidebar.radio("Optimize model for:").set_value("Size (Maximum Reduction)")
        
        # Run again to process these radio button changes
        app_test.run(timeout=30)

        # Assertions based on main_v2.py's logic for "Size (Maximum Reduction)"
        self.assertTrue(app_test.session_state.block_pruning)
        self.assertTrue(app_test.session_state.channel_pruning)
        self.assertTrue(app_test.session_state.quantization)
        self.assertEqual(app_test.session_state.quantization_type, "int8") # Per main_v2.py code
        self.assertEqual(app_test.session_state.block_pruning_ratio, 0.7)
        self.assertEqual(app_test.session_state.channel_pruning_ratio, 0.6)
        print(f"Session State after test: {app_test.session_state}\n")


class TestBasicModeSpeedBalanced(BaseUITest):
    def test_basic_mode_speed_balanced(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Automatic' mode with 'Speed (Balanced)' selected.")
        print("       Verifies correct session state settings for balanced speed optimization.")
        print("Expected: (Example) session_state.block_pruning = True, channel_pruning = True, quantization = True, quantization_type = 'float16'") # Adjust as per your app's logic
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Automatic")
        app_test.sidebar.radio("Optimize model for:").set_value("Speed (Balanced)")
        app_test.run(timeout=30)

        self.assertTrue(app_test.session_state.block_pruning)
        self.assertTrue(app_test.session_state.channel_pruning)
        self.assertTrue(app_test.session_state.quantization)
        self.assertEqual(app_test.session_state.quantization_type, "float16")
        self.assertEqual(app_test.session_state.block_pruning_ratio, 0.5)
        self.assertEqual(app_test.session_state.channel_pruning_ratio, 0.4)
        print(f"Session State after test: {app_test.session_state}\n")


class TestBasicModeAccuracyMinimalLoss(BaseUITest):
    def test_basic_mode_accuracy_minimal_loss(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Automatic' mode with 'Accuracy (Minimal Loss)' selected.")
        print("       Verifies correct session state settings for minimal accuracy loss optimization.")
        print("Expected: session_state.block_pruning = True, channel_pruning = False, quantization = True, quantization_type = 'dynamic', block_pruning_ratio = 0.3")
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Automatic")
        app_test.sidebar.radio("Optimize model for:").set_value("Accuracy (Minimal Loss)")
        app_test.run(timeout=30)

        self.assertTrue(app_test.session_state.block_pruning)
        self.assertFalse(app_test.session_state.channel_pruning)
        self.assertTrue(app_test.session_state.quantization)
        self.assertEqual(app_test.session_state.quantization_type, "dynamic")
        self.assertEqual(app_test.session_state.block_pruning_ratio, 0.3)
        print(f"Session State after test: {app_test.session_state}\n")


class TestAdvancedModeBlockOnly(BaseUITest):
    def test_advanced_mode_block_pruning_only(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Advanced' mode with only 'Block Level Pruning' selected.")
        print("       Verifies session state reflects this single selection.")
        print("Expected: session_state.block_pruning = True, others = False")
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Advanced")
        app_test.run(timeout=30) # Re-run to render Advanced options in sidebar

        app_test.sidebar.checkbox("Block Level Pruning").set_value(True)
        app_test.sidebar.checkbox("Channel Pruning").set_value(False)
        app_test.sidebar.checkbox("Knowledge Distillation").set_value(False)
        app_test.sidebar.checkbox("Quantization").set_value(False)
        app_test.run(timeout=30)

        self.assertTrue(app_test.session_state.block_pruning)
        self.assertFalse(app_test.session_state.channel_pruning)
        self.assertFalse(app_test.session_state.knowledge_distillation)
        self.assertFalse(app_test.session_state.quantization)
        print(f"Session State after test: {app_test.session_state}\n")


class TestAdvancedModeChannelOnly(BaseUITest):
    def test_advanced_mode_channel_pruning_only(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Advanced' mode with only 'Channel Pruning' selected.")
        print("       Verifies session state reflects this single selection.")
        print("Expected: session_state.channel_pruning = True, others = False")
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Advanced")
        app_test.run(timeout=30) 

        app_test.sidebar.checkbox("Block Level Pruning").set_value(False)
        app_test.sidebar.checkbox("Channel Pruning").set_value(True)
        app_test.sidebar.checkbox("Knowledge Distillation").set_value(False)
        app_test.sidebar.checkbox("Quantization").set_value(False)
        app_test.run(timeout=30)

        self.assertFalse(app_test.session_state.block_pruning)
        self.assertTrue(app_test.session_state.channel_pruning)
        self.assertFalse(app_test.session_state.knowledge_distillation)
        self.assertFalse(app_test.session_state.quantization)
        print(f"Session State after test: {app_test.session_state}\n")


class TestAdvancedModeKDOnly(BaseUITest):
    def test_advanced_mode_kd_only(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Advanced' mode with only 'Knowledge Distillation' selected.")
        print("       Verifies session state reflects this single selection.")
        print("Expected: session_state.knowledge_distillation = True, others = False")
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Advanced")
        app_test.run(timeout=30) 

        app_test.sidebar.checkbox("Block Level Pruning").set_value(False)
        app_test.sidebar.checkbox("Channel Pruning").set_value(False)
        app_test.sidebar.checkbox("Knowledge Distillation").set_value(True)
        app_test.sidebar.checkbox("Quantization").set_value(False)
        app_test.run(timeout=30)

        self.assertFalse(app_test.session_state.block_pruning)
        self.assertFalse(app_test.session_state.channel_pruning)
        self.assertTrue(app_test.session_state.knowledge_distillation)
        self.assertFalse(app_test.session_state.quantization)
        print(f"Session State after test: {app_test.session_state}\n")


class TestAdvancedModeQuantizationOnly(BaseUITest):
    def test_advanced_mode_quantization_only(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Advanced' mode with only 'Quantization' selected.")
        print("       Verifies session state reflects this single selection.")
        print("Expected: session_state.quantization = True, others = False")
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Advanced")
        app_test.run(timeout=30) 

        app_test.sidebar.checkbox("Block Level Pruning").set_value(False)
        app_test.sidebar.checkbox("Channel Pruning").set_value(False)
        app_test.sidebar.checkbox("Knowledge Distillation").set_value(False)
        app_test.sidebar.checkbox("Quantization").set_value(True)
        app_test.run(timeout=30)

        self.assertFalse(app_test.session_state.block_pruning)
        self.assertFalse(app_test.session_state.channel_pruning)
        self.assertFalse(app_test.session_state.knowledge_distillation)
        self.assertTrue(app_test.session_state.quantization)
        print(f"Session State after test: {app_test.session_state}\n")


class TestAdvancedModeAllMethods(BaseUITest):
    def test_advanced_mode_all_methods_enabled(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Advanced' mode with all optimization methods selected.")
        print("       Verifies session state reflects all selections being true.")
        print("Expected: session_state.block_pruning = True, channel_pruning = True, knowledge_distillation = True, quantization = True")
        print("Actual (Simulated): Would check app_test.session_state after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Advanced")
        app_test.run(timeout=30) 

        app_test.sidebar.checkbox("Block Level Pruning").set_value(True)
        app_test.sidebar.checkbox("Channel Pruning").set_value(True)
        app_test.sidebar.checkbox("Knowledge Distillation").set_value(True)
        app_test.sidebar.checkbox("Quantization").set_value(True)
        app_test.run(timeout=30)

        self.assertTrue(app_test.session_state.block_pruning)
        self.assertTrue(app_test.session_state.channel_pruning)
        self.assertTrue(app_test.session_state.knowledge_distillation)
        self.assertTrue(app_test.session_state.quantization)
        print(f"Session State after test: {app_test.session_state}\n")


class TestAdvancedModeNoneSelected(BaseUITest):
    def test_no_optimization_method_selected(self):
        print(f"\n--- Test: {self._testMethodName} ---")
        print("Logic: Test 'Advanced' mode with no optimization methods selected, then clicking 'Optimize Model'.")
        print("       Verifies that an error message is displayed.")
        print("Expected: An st.error element with text 'Please select at least one optimization method' appears.")
        print("Actual (Simulated): Would check app_test.error for the message after interactions.")

        app_test = AppTest.from_file("main_v2.py")
        app_test.run(timeout=30) 

        app_test.sidebar.radio("Select Mode:").set_value("Advanced")
        app_test.run(timeout=30) 

        app_test.sidebar.checkbox("Block Level Pruning").set_value(False)
        app_test.sidebar.checkbox("Channel Pruning").set_value(False)
        app_test.sidebar.checkbox("Knowledge Distillation").set_value(False)
        app_test.sidebar.checkbox("Quantization").set_value(False)
        
        # The "Optimize Model" button is in the main content area, not sidebar
        app_test.button("Optimize Model").click() 
        app_test.run(timeout=30)

        self.assertGreater(len(app_test.error), 0, "Expected an error message, but none was found.")
        self.assertIn("Please select at least one optimization method", app_test.error[0].value)
        print(f"Error message found: {app_test.error[0].value}")
        print(f"Session State after test: {app_test.session_state}\n")

if __name__ == '__main__':
    unittest.main()