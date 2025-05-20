import pytest
from unittest.mock import patch, MagicMock, mock_open
import torch
import torch.nn as nn
import os
import tempfile

# Import the Streamlit application script.
# Ensure main_v2.py is in the same directory or PYTHONPATH
import main_v2 as app_to_test

# Dummy model for testing purposes
class DummyModel(nn.Module):
    def __init__(self, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Create a structure that identify_model_blocks can parse
            # It needs a named module like 'blocks.X' which is an nn.Module
            block_module = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
            self.add_module(f"blocks.{i}", block_module) # Correctly adds 'blocks.0', 'blocks.1'
        self.classifier = nn.Linear(10, 1)

    def forward(self, x):
        # Simplified forward pass
        for i in range(len(self.blocks)): # Accessing via index on ModuleList is not how named_modules works
            x = getattr(self, f"blocks.{i}")(x) # This ensures named modules are used
        return self.classifier(x)

@pytest.fixture
def dummy_model_path():
    """Creates a temporary dummy PyTorch model file."""
    model = DummyModel(num_blocks=3)
    # Use a real temporary file for torch.save and subsequent load simulation
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        torch.save(model, tmp.name)
        # tmp.flush() # Ensure data is written
        return tmp.name

@pytest.fixture
def dummy_teacher_model_path():
    """Creates a temporary dummy PyTorch teacher model file."""
    model = DummyModel(num_blocks=3) # Teacher can be same/different
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        torch.save(model, tmp.name)
        return tmp.name

@pytest.fixture(autouse=True)
def mock_streamlit_elements(mocker):
    """Mocks all streamlit calls used in main_v2.py."""
    st_mocks = {
        'set_page_config': mocker.patch('streamlit.set_page_config'),
        'markdown': mocker.patch('streamlit.markdown'),
        'sidebar': mocker.patch('streamlit.sidebar', new_callable=MagicMock),
        'file_uploader': mocker.patch('streamlit.file_uploader'),
        'radio': mocker.patch('streamlit.radio'),
        'checkbox': mocker.patch('streamlit.checkbox'),
        'slider': mocker.patch('streamlit.slider'),
        'expander': mocker.patch('streamlit.expander', new_callable=MagicMock),
        'tabs': mocker.patch('streamlit.tabs'),
        'button': mocker.patch('streamlit.button'),
        'progress': mocker.patch('streamlit.progress'),
        'empty': mocker.patch('streamlit.empty'),
        'spinner': mocker.patch('streamlit.spinner', new_callable=MagicMock),
        'success': mocker.patch('streamlit.success'),
        'error': mocker.patch('streamlit.error'),
        'info': mocker.patch('streamlit.info'),
        'metric': mocker.patch('streamlit.metric'),
        'subheader': mocker.patch('streamlit.subheader'),
        'columns': mocker.patch('streamlit.columns'),
        'download_button': mocker.patch('streamlit.download_button'),
        'container': mocker.patch('streamlit.container', new_callable=MagicMock),
        # Important: Use the actual session_state object from the imported app module
        # This allows the test to modify the same session_state the app's code sees.
        'session_state': app_to_test.st.session_state
    }

    # Configure context manager mocks
    for mock_name in ['sidebar', 'expander', 'spinner', 'container']:
        mock_obj = st_mocks[mock_name]
        mock_obj.return_value.__enter__ = MagicMock(return_value=None)
        mock_obj.return_value.__exit__ = MagicMock(return_value=False)
        if mock_name == 'sidebar': # if st.sidebar: ... then st.sidebar.radio()
             mocker.patch.object(app_to_test.st.sidebar, 'radio', st_mocks['radio'])
             mocker.patch.object(app_to_test.st.sidebar, 'subheader', st_mocks['subheader'])
             mocker.patch.object(app_to_test.st.sidebar, 'file_uploader', st_mocks['file_uploader'])
             mocker.patch.object(app_to_test.st.sidebar, 'markdown', st_mocks['markdown'])
             mocker.patch.object(app_to_test.st.sidebar, 'checkbox', st_mocks['checkbox'])


    # Mock st.tabs to return a list of mock tab objects
    mock_tab = MagicMock()
    mock_tab.__enter__ = MagicMock(return_value=None)
    mock_tab.__exit__ = MagicMock(return_value=False)
    st_mocks['tabs'].return_value = [mock_tab] * 5 # Provide enough mock tabs

    # Mock st.empty() to return an object with a text method
    mock_empty_instance = MagicMock()
    st_mocks['empty'].return_value = mock_empty_instance

    # Clear session state before each test for isolation
    # and set initial defaults as in the app
    app_to_test.st.session_state.clear()
    app_to_test.st.session_state.previous_mode = "Automatic"
    app_to_test.st.session_state.adv_tab_selected = False

    return st_mocks

@pytest.fixture(autouse=True)
def mock_app_dependencies(mocker):
    """Mocks dependencies of main_v2.py (torch utils, pruning, quantization funcs)."""
    mocker.patch('main_v2.count_net_flops', return_value=1000000)
    mocker.patch('main_v2.count_peak_activation_size', return_value=500000)
    mocker.patch('main_v2.get_model_size', return_value=10.0) # Original size in MB
    mocker.patch('main_v2.main_pruning_loop', side_effect=lambda model, **kwargs: model) # Returns the model
    mocker.patch('main_v2.knowledge_distillation_prune', side_effect=lambda teacher_model, student_model, **kwargs: student_model)
    mocker.patch('main_v2.quantize_model', return_value=None) # Modifies/saves files
    mocker.patch('main_v2.get_tflite_model_size', return_value=2.0) # Quantized size in MB
    mocker.patch('main_v2.test_model', return_value=90.0) # Mocked accuracy

    # File system mocks
    mocker.patch('os.path.exists', return_value=True) # Assume files exist for cleanup checks
    mocker.patch('os.unlink', return_value=None) # Mock unlink to prevent errors

    # Mock torch.load to return a new instance of DummyModel each time it's called
    # This ensures that different parts of the code (e.g., loading main model vs. teacher model)
    # get distinct model objects if needed, though for this test, one type is fine.
    mocker.patch('torch.load', side_effect=lambda path, map_location, **kwargs: DummyModel(num_blocks=3))
    mocker.patch('torch.save', return_value=None)

    # Mock random.randint used for accuracy simulation
    mocker.patch('random.randint', side_effect=lambda a,b: (a+b)//2) # Predictable random value

    # Patch DEVICE to be CPU for tests
    mocker.patch('main_v2.DEVICE', torch.device('cpu'))


def execute_button_click_logic(app_module, model_obj, block_ratios_map, temp_model_path, temp_teacher_model_path=None):
    """
    This function encapsulates and executes the logic found within the
    `if st.button("Optimize Model")` block in main_v2.py.
    It directly uses the `app_module.st.session_state` and mocked functions.
    Args:
        app_module: The imported main_v2.py module (aliased as app_to_test).
        model_obj: The mock model object that would have been loaded.
        block_ratios_map: The precalculated block pruning ratios.
        temp_model_path: Path to the (dummy) uploaded model file.
        temp_teacher_model_path: Path to the (dummy) teacher model file if KD is active.
    """
    # This is a simplified adaptation of the button click logic from main_v2.py
    # It assumes st.session_state is already configured for the desired test scenario.

    # Check if any optimization method is selected
    if not (app_module.st.session_state.block_pruning or app_module.st.session_state.channel_pruning or
            app_module.st.session_state.knowledge_distillation or app_module.st.session_state.quantization):
        app_module.st.error("Please select at least one optimization method before proceeding.")
        return "error_no_method_selected"

    # Check for teacher model if KD is selected
    if app_module.st.session_state.knowledge_distillation and (app_module.st.session_state.get('teacher_model_uploader') is None):
        app_module.st.error("Please upload a Teacher Model for Knowledge Distillation.")
        return "error_no_teacher_model"

    with app_module.st.spinner("Optimizing your model..."): # Uses mocked spinner
        dummy_input_shape = (1, 1, 96, 96) # From main_v2.py (ensure this matches DummyModel or mock robustly)
        # For DummyModel expecting (batch, 10), this input won't match.
        # However, count_net_flops/peak_activation_size are mocked, so it might not matter for them.
        # If these mocks were more complex or real functions were used, input would be critical.
        # For this test, we'll assume the mocks handle it.

        original_size_mb = app_module.get_model_size(model_obj)
        original_flops = app_module.count_net_flops(model_obj, dummy_input_shape)
        original_acc = random.randint(80, 95) # Uses mocked random.randint

        current_model_for_opt = model_obj # Start with the loaded model

        pruning_type = ""
        if app_module.st.session_state.block_pruning and app_module.st.session_state.channel_pruning:
            pruning_type = "BOTH"
        elif app_module.st.session_state.block_pruning:
            pruning_type = "BLOCK"
        elif app_module.st.session_state.channel_pruning:
            pruning_type = "UNIFORM"

        if app_module.st.session_state.block_pruning or app_module.st.session_state.channel_pruning:
            app_module.st.empty().text("Applying pruning...") # Use mocked st.empty
            current_model_for_opt = app_module.main_pruning_loop(
                model=current_model_for_opt,
                block_level_dict=block_ratios_map if app_module.st.session_state.block_pruning else {},
                uniform_pruning_ratio=app_module.st.session_state.channel_pruning_ratio if app_module.st.session_state.channel_pruning else 0.0,
                block_fine_tune_epochs=app_module.st.session_state.get('block_fine_tune_epochs', 0) if app_module.st.session_state.get('block_fine_tune') else 0,
                channel_fine_tune_epochs=app_module.st.session_state.get('channel_fine_tune_epochs', 0) if app_module.st.session_state.get('channel_fine_tune') else 0,
                device=app_module.DEVICE,
                type=pruning_type
            )
            app_module.st.success("Pruning complete.")

        distilled_model = current_model_for_opt

        if app_module.st.session_state.knowledge_distillation:
            app_module.st.empty().text("Applying knowledge distillation...")
            # In a real test for KD, you'd ensure 'teacher_model_uploader' is a mock file object
            # and that torch.load is called for its path.
            teacher_model_file_obj = app_module.st.session_state.get('teacher_model_uploader') # This should be a mock file
            
            # Simulate loading teacher model
            # The actual torch.load is mocked to return a DummyModel instance.
            # The NamedTemporaryFile for teacher model is handled outside if used.
            teacher_model_loaded = app_module.torch.load(temp_teacher_model_path, map_location=app_module.DEVICE, weights_only=False)

            distilled_model = app_module.knowledge_distillation_prune(
                teacher_model=teacher_model_loaded,
                student_model=distilled_model,
                num_epochs=app_module.st.session_state.distillation_epochs,
                device=app_module.DEVICE
            )
            app_module.st.success("Knowledge Distillation complete.")

        final_model_after_pruning_distillation = distilled_model
        # Metrics after pruning/distillation
        pruned_size_mb = app_module.get_model_size(final_model_after_pruning_distillation)
        pruned_flops = app_module.count_net_flops(final_model_after_pruning_distillation, dummy_input_shape)
        if app_module.st.session_state.knowledge_distillation:
            pruned_acc = max(original_acc - random.randint(0, 5), 0)
        else:
            pruned_acc = max(original_acc - random.randint(0, 10), 0)


        # Mock saving of intermediate models
        pruned_state_dict_path_val = os.path.join(tempfile.gettempdir(), "pruned_model_state_dict.pth")
        app_module.torch.save(final_model_after_pruning_distillation.state_dict(), pruned_state_dict_path_val)
        model_before_quant_path_val = os.path.join(tempfile.gettempdir(), "model_before_quant.pth")
        app_module.torch.save(final_model_after_pruning_distillation, model_before_quant_path_val)

        quantized_model_path_local = None
        quantized_size_local = None
        if app_module.st.session_state.quantization:
            app_module.st.empty().text(f"Applying {app_module.st.session_state.quantization_type} quantization...")
            quantized_model_path_local = os.path.join(tempfile.gettempdir(), "quantized_model.tflite")
            app_module.quantize_model(model_before_quant_path_val, quantized_model_path_local, "person_detection_validation", app_module.st.session_state.quantization_type)
            quantized_size_local = app_module.get_tflite_model_size(quantized_model_path_local)
            app_module.st.success("Quantization complete.")

        app_module.st.empty().text("Optimization complete!")
        # Metrics display (simplified check - ensure st.metric is called)
        app_module.st.metric("Size", f"{original_size_mb:.2f} MB") # Example call

        # Cleanup (uses mocked os.unlink and os.path.exists)
        app_module.os.unlink(temp_model_path) # Original uploaded model temp path
        if app_module.os.path.exists(pruned_state_dict_path_val): app_module.os.unlink(pruned_state_dict_path_val)
        if app_module.os.path.exists(model_before_quant_path_val): app_module.os.unlink(model_before_quant_path_val)
        if quantized_model_path_local and app_module.os.path.exists(quantized_model_path_local): app_module.os.unlink(quantized_model_path_local)

    return "success"


def test_identify_model_blocks_logic():
    """Tests the identify_model_blocks function directly."""
    model_with_blocks = DummyModel(num_blocks=2)
    # Expected: ['blocks.0', 'blocks.1'] because DummyModel creates them this way.
    identified = app_to_test.identify_model_blocks(model_with_blocks)
    assert identified == ['blocks.0', 'blocks.1']

    model_no_blocks = nn.Linear(10,1) # A model with no 'blocks.X' structure
    identified_empty = app_to_test.identify_model_blocks(model_no_blocks)
    assert identified_empty == []


def test_full_optimization_flow_automatic_size(
    mock_streamlit_elements, mock_app_dependencies, dummy_model_path # pytest fixtures
):
    """
    Tests the "Automatic - Size (Maximum Reduction)" optimization path.
    """
    st_mocks = mock_streamlit_elements # Get the dictionary of mocks

    # --- Simulate User Input and App State Setup ---
    # 1. Simulate file upload: st.file_uploader will be called by the app's sidebar logic.
    #    We configure its mock to return a dummy file object.
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "dummy_model.pth"
    # getvalue() needs to return bytes
    with open(dummy_model_path, 'rb') as f_bytes:
        dummy_bytes_content = f_bytes.read()
    mock_uploaded_file.getvalue = MagicMock(return_value=dummy_bytes_content)
    st_mocks['file_uploader'].return_value = mock_uploaded_file # For the main model

    # 2. Simulate radio button selections for "Automatic" mode and "Size" profile.
    #    The app's sidebar logic will call st.radio().
    st_mocks['radio'].side_effect = [
        "Automatic",  # First call to st.radio (mode selection)
        "Size (Maximum Reduction)"  # Second call (profile selection)
    ]

    # 3. Re-run the app's setup logic (conceptually)
    #    In a real Streamlit app, changing a widget re-runs the script.
    #    Here, we need to manually trigger the session_state updates that would happen.
    #    The app's own logic:
    #    `if optimization_mode == "Automatic":`
    #    `   if optimize_for == "Size (Maximum Reduction)":`
    #    `       st.session_state.block_pruning = True ...`
    #    We will set these directly based on the expected outcome of mocked radio calls.

    app_to_test.st.session_state.optimization_mode = "Automatic" # Would be set by first radio
    app_to_test.st.session_state.optimize_for = "Size (Maximum Reduction)" # By second

    # Apply "Size (Maximum Reduction)" profile settings to session_state
    app_to_test.st.session_state.block_pruning = True
    app_to_test.st.session_state.channel_pruning = True
    app_to_test.st.session_state.knowledge_distillation = False # Key for this profile
    app_to_test.st.session_state.quantization = True
    app_to_test.st.session_state.quantization_type = "int8"
    app_to_test.st.session_state.block_pruning_ratio = 0.7
    app_to_test.st.session_state.channel_pruning_ratio = 0.6
    app_to_test.st.session_state.block_fine_tune = True
    app_to_test.st.session_state.channel_fine_tune = True
    app_to_test.st.session_state.block_fine_tune_epochs = 3
    app_to_test.st.session_state.channel_fine_tune_epochs = 3
    app_to_test.st.session_state.adv_tab_selected = False # Automatic mode

    # --- Prepare for Button Click Logic Execution ---
    # The model that torch.load (mocked) will return via the fixture
    # The actual `torch.load` call happens inside the button click logic if we extract it.
    # `dummy_model_path` is the path to the temp file created by the fixture.
    # Our mocked `torch.load` will be called with this path.
    loaded_model_obj = app_to_test.torch.load(dummy_model_path, map_location=app_to_test.DEVICE, weights_only=False) # Simulate load

    # Calculate block_pruning_ratios as the script would in the "Automatic" section
    # `blocks = identify_model_blocks(model)`
    # `for block in blocks: block_pruning_ratios[block] = st.session_state.block_pruning_ratio`
    identified_blocks = app_to_test.identify_model_blocks(loaded_model_obj)
    current_block_pruning_ratios = {}
    if app_to_test.st.session_state.block_pruning:
        ratio = app_to_test.st.session_state.block_pruning_ratio
        for block_name in identified_blocks:
            current_block_pruning_ratios[block_name] = ratio

    # --- Execute the button click logic ---
    result = execute_button_click_logic(app_to_test, loaded_model_obj, current_block_pruning_ratios, dummy_model_path)
    assert result == "success"

    # --- Assertions ---
    mock_app_dependencies['main_pruning_loop'].assert_called_once()
    call_kwargs = mock_app_dependencies['main_pruning_loop'].call_args[1]
    assert call_kwargs['type'] == "BOTH"
    assert call_kwargs['block_level_dict'] == current_block_pruning_ratios
    assert call_kwargs['uniform_pruning_ratio'] == 0.6
    assert call_kwargs['block_fine_tune_epochs'] == 3
    assert call_kwargs['channel_fine_tune_epochs'] == 3

    mock_app_dependencies['quantize_model'].assert_called_once()
    quant_call_args, quant_call_kwargs = mock_app_dependencies['quantize_model'].call_args
    assert quant_call_args[2] == "person_detection_validation" # dataset_name from app
    assert quant_call_args[3] == "int8" # quantization_type

    mock_app_dependencies['knowledge_distillation_prune'].assert_not_called()

    st_mocks['success'].assert_any_call("Pruning complete.")
    st_mocks['success'].assert_any_call("Quantization complete.")
    st_mocks['success'].assert_any_call("Optimization complete!") # Check final message
    st_mocks['error'].assert_not_called() # No errors expected in this path

    # Check cleanup of original model temp file
    mock_app_dependencies['os.unlink'].assert_any_call(dummy_model_path)

    # Cleanup the temp file created by the fixture
    os.unlink(dummy_model_path)


def test_error_no_optimization_method_selected(
    mock_streamlit_elements, mock_app_dependencies, dummy_model_path
):
    # Set session state so no optimization method is chosen
    app_to_test.st.session_state.block_pruning = False
    app_to_test.st.session_state.channel_pruning = False
    app_to_test.st.session_state.knowledge_distillation = False
    app_to_test.st.session_state.quantization = False

    # Model object that would be loaded
    loaded_model_obj = app_to_test.torch.load(dummy_model_path, map_location=app_to_test.DEVICE, weights_only=False)

    result = execute_button_click_logic(app_to_test, loaded_model_obj, {}, dummy_model_path)
    assert result == "error_no_method_selected"
    mock_streamlit_elements['error'].assert_called_once_with("Please select at least one optimization method before proceeding.")

    os.unlink(dummy_model_path)


def test_error_kd_selected_but_no_teacher_model_uploaded(
    mock_streamlit_elements, mock_app_dependencies, dummy_model_path
):
    app_to_test.st.session_state.block_pruning = False # Minimal other ops
    app_to_test.st.session_state.channel_pruning = False
    app_to_test.st.session_state.knowledge_distillation = True # KD is ON
    app_to_test.st.session_state.quantization = False
    # Crucially, st.session_state['teacher_model_uploader'] is None (or not set)

    loaded_model_obj = app_to_test.torch.load(dummy_model_path, map_location=app_to_test.DEVICE, weights_only=False)

    result = execute_button_click_logic(app_to_test, loaded_model_obj, {}, dummy_model_path)
    assert result == "error_no_teacher_model"
    mock_streamlit_elements['error'].assert_called_once_with("Please upload a Teacher Model for Knowledge Distillation.")

    os.unlink(dummy_model_path)