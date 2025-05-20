# integration_test_app.py
import os
import tempfile
import torch
import unittest
from unittest.mock import MagicMock, patch
from streamlit.testing.v1 import AppTest

# Define a minimal compatible test model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 1, kernel_size=3, padding=1)
            )
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class TestFullOptimizationWorkflow(unittest.TestCase):
    def setUp(self):
        # Save a compatible model to temp path
        self.model = SimpleModel()
        self.model_path = os.path.join(tempfile.gettempdir(), "test_model.pth")
        torch.save(self.model, self.model_path)

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    def test_automatic_size_optimization(self):
        app = AppTest.from_file("main_v2.py")

        # First run to build UI
        app.run(timeout=10000)

        # Upload model AFTER rendering the UI
        with open(self.model_path, "rb") as f:
            app.sidebar.file_uploader(label="Upload PyTorch model (.pth)").set_value({
                "name": "test_model.pth",
                "type": "application/octet-stream",
                "data": f.read()
            })

        # Set other sidebar inputs
        app.sidebar.radio("Select Mode:").set_value("Automatic")
        app.sidebar.radio("Optimize model for:").set_value("Size (Maximum Reduction)")

        # Click optimize and run again
        app.button("Optimize Model").click()
        app.run(timeout=60000)

        # Verify session state updated
        assert app.session_state.block_pruning is True
        assert app.session_state.quantization_type == "int8"

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


if __name__ == "__main__":
    unittest.main()
