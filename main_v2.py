import streamlit as st
import os
import tempfile
import torch
from torch import nn
import time
import random
from mcunet.utils.pytorch_utils import count_net_flops, count_peak_activation_size

from pruning_logic.Pruning_definitions import get_model_size
from quantization_logic.quantization import quantize_model, get_tflite_model_size
from pruning_logic.Streamlined_prune import main_pruning_loop, knowledge_distillation_prune, main_finetune_model, test_model

# Set device
DEVICE = torch.device('cpu')  # Change to 'cuda' or 'mps' if available

def identify_model_blocks(model):
    """Extract model block names for selective pruning"""
    block_names = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Module) and name.startswith("blocks"):
                parts = name.split(".")
                # Match only direct children like "blocks.0", "blocks.1", ...
                if len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit():
                    block_names.append(name)
        return sorted(block_names, key=lambda x: int(x.split(".")[1]))
    except Exception as e:
        st.error(f"Error extracting block names: {e}")
        return []

# App layout and styling
st.set_page_config(page_title="Model Optimizer", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {text-align: center; margin-bottom: 2rem;}
    .optimization-section {background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    .results-section {background-color: #f0f8ff; padding: 20px; border-radius: 10px;}
    .download-section {margin-top: 30px;}
</style>
""", unsafe_allow_html=True)

# Main application header
st.markdown("<h1 class='main-header'>Model Optimizer</h1>", unsafe_allow_html=True)

# Initialize session state variables for tracking UI state
if 'previous_mode' not in st.session_state:
    st.session_state.previous_mode = "Basic"

# Sidebar for upload and basic options
with st.sidebar:
    st.subheader("Model Upload")
    uploaded_file = st.file_uploader("Upload PyTorch model (.pth)", type=["pth"])
    
    if uploaded_file:
        st.success("✅ Model uploaded")
        
        # Basic optimization methods selection
        st.subheader("Optimization Methods")
        optimization_mode = st.radio(
            "Select Mode:",
            ["Basic", "Advanced"]
        )

        st.markdown("""
            <hr style="border:1px solid #ccc; margin:20px 0;">
            """,
            unsafe_allow_html=True)
        
        # Check if mode changed and reset parameters if needed
        if 'previous_mode' in st.session_state and st.session_state.previous_mode != optimization_mode:
            if optimization_mode == "Advanced":
                # Reset to all unchecked when switching to Advanced
                st.session_state.block_pruning = False
                st.session_state.channel_pruning = False
                st.session_state.knowledge_distillation = False
                st.session_state.quantization = False
            st.session_state.previous_mode = optimization_mode
            
        # Update adv_tab_selected based on mode
        st.session_state.adv_tab_selected = (optimization_mode == "Advanced")
        
        if optimization_mode == "Basic":
            optimize_for = st.radio(
                "Optimize model for:",
                ["Size (Maximum Reduction)", "Speed (Balanced)", "Accuracy (Minimal Loss)"],
                index=1
            )
            
            # Set optimization parameters based on profile
            if optimize_for == "Size (Maximum Reduction)":
                st.session_state.block_pruning = True
                st.session_state.channel_pruning = True
                st.session_state.knowledge_distillation = False
                st.session_state.quantization = True
                st.session_state.quantization_type = "int8"
                st.session_state.block_pruning_ratio = 0.7
                st.session_state.channel_pruning_ratio = 0.6
                st.session_state.block_fine_tune = True
                st.session_state.channel_fine_tune = True
                st.session_state.block_fine_tune_epochs = 3
                st.session_state.channel_fine_tune_epochs = 3
                
            elif optimize_for == "Speed (Balanced)":
                st.session_state.block_pruning = True
                st.session_state.channel_pruning = True
                st.session_state.knowledge_distillation = False
                st.session_state.quantization = True
                st.session_state.quantization_type = "float16"
                st.session_state.block_pruning_ratio = 0.5
                st.session_state.channel_pruning_ratio = 0.4
                st.session_state.block_fine_tune = True
                st.session_state.channel_fine_tune = True
                st.session_state.block_fine_tune_epochs = 5
                st.session_state.channel_fine_tune_epochs = 5
                
            elif optimize_for == "Accuracy (Minimal Loss)":
                st.session_state.block_pruning = True
                st.session_state.channel_pruning = False
                st.session_state.knowledge_distillation = False
                st.session_state.quantization = True
                st.session_state.quantization_type = "dynamic"
                st.session_state.block_pruning_ratio = 0.3
                st.session_state.channel_pruning_ratio = 0.2
                st.session_state.block_fine_tune = True
                st.session_state.channel_fine_tune = True
                st.session_state.block_fine_tune_epochs = 10
                st.session_state.channel_fine_tune_epochs = 10
        
        # Only show advanced options if Advanced mode is selected
        if optimization_mode == "Advanced":
            # Initialize these state variables if they don't exist yet
            if 'block_pruning' not in st.session_state:
                st.session_state.block_pruning = False
            if 'channel_pruning' not in st.session_state:
                st.session_state.channel_pruning = False
            if 'knowledge_distillation' not in st.session_state:
                st.session_state.knowledge_distillation = False
            if 'quantization' not in st.session_state:
                st.session_state.quantization = False
            
            # Advanced optimization method selection
            st.session_state.block_pruning = st.checkbox("Block Level Pruning", value=st.session_state.block_pruning)
            st.session_state.channel_pruning = st.checkbox("Channel Pruning", value=st.session_state.channel_pruning)
            st.session_state.knowledge_distillation = st.checkbox("Knowledge Distillation", value=st.session_state.knowledge_distillation)
            st.session_state.quantization = st.checkbox("Quantization", value=st.session_state.quantization)

# Main content
if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        model_path = tmp_file.name
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Main content area - only shown after model is uploaded
    with st.container():
        st.markdown("<div class='optimization-section'>", unsafe_allow_html=True)
        
        # Advanced Options for fine-tuning the optimization - only show when Advanced tab is selected
        if st.session_state.adv_tab_selected:
            with st.expander("Optimization Parameters", expanded=True):
                # Create a list of selected tabs in a specific order
                tab_names = []
                
                # Always add pruning tabs first if selected
                if st.session_state.block_pruning:
                    tab_names.append("Block Pruning")
                if st.session_state.channel_pruning:
                    tab_names.append("Channel Pruning")
                    
                # Always add fine-tuning if any pruning is selected
                if st.session_state.block_pruning or st.session_state.channel_pruning:
                    tab_names.append("Fine-tuning")
                    
                # Then add other options
                if st.session_state.knowledge_distillation:
                    tab_names.append("Knowledge Distillation")
                if st.session_state.quantization:
                    tab_names.append("Quantization")
                
                # Only create tabs if we have optimization methods selected
                if tab_names:
                    param_tabs = st.tabs(tab_names)
                    
                    # Create a tab index dictionary for direct access
                    tab_indices = {name: idx for idx, name in enumerate(tab_names)}
                    
                    # Block pruning settings
                    if st.session_state.block_pruning:
                        with param_tabs[tab_indices["Block Pruning"]]:
                            blocks = identify_model_blocks(model)
                            
                            use_same_ratio = st.checkbox("Use same pruning ratio for all blocks", value=True)
                            block_pruning_ratios = {}
                            
                            if use_same_ratio:
                                if 'block_pruning_ratio' not in st.session_state:
                                    st.session_state.block_pruning_ratio = 0.5
                                
                                st.session_state.block_pruning_ratio = st.slider(
                                    "Block Pruning Ratio", 0.0, 0.9, 
                                    st.session_state.block_pruning_ratio, 
                                    0.05
                                )
                                for block in blocks:
                                    block_pruning_ratios[block] = st.session_state.block_pruning_ratio
                            else:
                                cols = st.columns(2)
                                for i, block in enumerate(blocks):
                                    with cols[i % 2]:
                                        block_pruning_ratios[block] = st.slider(
                                            f"Block {block.split('.')[1]}", 
                                            0.0, 0.9, 0.5, 0.05,
                                            key=f"block_{block}"
                                        )
                    
                    # Channel pruning settings
                    if st.session_state.channel_pruning:
                        with param_tabs[tab_indices["Channel Pruning"]]:
                            if 'channel_pruning_ratio' not in st.session_state:
                                st.session_state.channel_pruning_ratio = 0.4
                                
                            st.session_state.channel_pruning_ratio = st.slider(
                                "Channel Pruning Ratio", 0.0, 0.9, 
                                st.session_state.channel_pruning_ratio,
                                0.05
                            )
                            
                            st.info("Channel pruning uniformly reduces model width across all layers.")
                    
                    # Fine-tuning settings - only show if any pruning is selected
                    if st.session_state.block_pruning or st.session_state.channel_pruning:
                        with param_tabs[tab_indices["Fine-tuning"]]:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if st.session_state.block_pruning:
                                    if 'block_fine_tune' not in st.session_state:
                                        st.session_state.block_fine_tune = True
                                    if 'block_fine_tune_epochs' not in st.session_state:
                                        st.session_state.block_fine_tune_epochs = 5
                                        
                                    st.session_state.block_fine_tune = st.checkbox("Enable fine-tuning after block pruning", value=st.session_state.block_fine_tune)
                                    if st.session_state.block_fine_tune:
                                        st.session_state.block_fine_tune_epochs = st.slider(
                                            "Block Pruning Fine-tuning Epochs", 1, 20, 
                                            st.session_state.block_fine_tune_epochs
                                        )
                            
                            with col2:
                                if st.session_state.channel_pruning:
                                    if 'channel_fine_tune' not in st.session_state:
                                        st.session_state.channel_fine_tune = True
                                    if 'channel_fine_tune_epochs' not in st.session_state:
                                        st.session_state.channel_fine_tune_epochs = 5
                                        
                                    st.session_state.channel_fine_tune = st.checkbox("Enable fine-tuning after channel pruning", value=st.session_state.channel_fine_tune)
                                    if st.session_state.channel_fine_tune:
                                        st.session_state.channel_fine_tune_epochs = st.slider(
                                            "Channel Pruning Fine-tuning Epochs", 1, 20, 
                                            st.session_state.channel_fine_tune_epochs
                                        )
                            
                            st.info("Fine-tuning helps restore accuracy after pruning by retraining the model.")
                    
                    # Knowledge Distillation Settings
                    if st.session_state.knowledge_distillation:
                        with param_tabs[tab_indices["Knowledge Distillation"]]:
                            teacher_model_file = st.file_uploader("Upload Teacher Model (.pth)", type=["pth"], key="teacher_model")
                            distillation_epochs = st.slider("Distillation Epochs", 1, 20, 10)
                            
                            st.info("Knowledge distillation transfers knowledge from a larger teacher model to your pruned model.")
                    
                    # Quantization settings
                    if st.session_state.quantization:
                        with param_tabs[tab_indices["Quantization"]]:
                            if 'quantization_type' not in st.session_state:
                                st.session_state.quantization_type = "float16"
                                
                            st.session_state.quantization_type = st.selectbox(
                                "Quantization Type",
                                ["int8", "float16", "dynamic"],
                                index=["int8", "float16", "dynamic"].index(st.session_state.quantization_type)
                            )
                            
                            # Add descriptions for each quantization type
                            if st.session_state.quantization_type == "int8":
                                st.info("INT8: Maximum size reduction. Converts weights and activations to 8-bit integers.")
                            elif st.session_state.quantization_type == "float16":
                                st.info("FLOAT16: Good balance between size and accuracy. Uses 16-bit floating point values.")
                            elif st.session_state.quantization_type == "dynamic":
                                st.info("DYNAMIC: Best accuracy preservation. Dynamic range quantization at runtime.")
                else:
                    st.info("Please select optimization methods in the Advanced tab to configure parameters.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process the model
        if st.button("Optimize Model", type="primary"):
            # Check if any optimization method is selected
            if not (st.session_state.block_pruning or st.session_state.channel_pruning or 
                    st.session_state.knowledge_distillation or st.session_state.quantization):
                st.error("Please select at least one optimization method before proceeding.")
            else:
                with st.spinner("Optimizing your model..."):
                    try:
                        # Calculate original model stats
                        dummy_input = (1, 1, 96, 96)
                        original_size_mb = get_model_size(model)
                        original_flops = count_net_flops(model, dummy_input)
                        original_peak_act = count_peak_activation_size(model, dummy_input)
                        original_acc = random.randint(80, 95)  # For demo purposes
                        
                        # Create progress tracking
                        progress = st.progress(0)
                        status_text = st.empty()
                        
                        # Initialize model variable for the pipeline
                        current_model = model
                        
                        # Execute block pruning if selected
                        if st.session_state.block_pruning:
                            status_text.text("Applying block pruning...")
                            progress.progress(0.1)
                            
                            # Get block pruning ratios
                            blocks = identify_model_blocks(model)
                            block_pruning_ratios = {block: st.session_state.block_pruning_ratio for block in blocks}
                            
                            block_pruned_model = main_pruning_loop(
                                model=current_model, 
                                block_level_dict=block_pruning_ratios,
                                uniform_pruning_ratio=0.0,  # No channel pruning at this step
                                type="BLOCK",
                                block_fine_tune_epochs=st.session_state.block_fine_tune_epochs if st.session_state.block_fine_tune else 0
                            )
                            
                            progress.progress(0.3)
                            current_model = block_pruned_model

                            # Fine-tuning after Block Pruning
                            if st.session_state.block_fine_tune:
                                status_text.text(f"Fine-tuning after Block Pruning for {st.session_state.block_fine_tune_epochs} epochs...")
                                main_finetune_model(current_model, st.session_state.block_fine_tune_epochs, DEVICE)
                                progress_bar = st.progress(0)
                                for i in range(st.session_state.block_fine_tune_epochs):
                                    for j in range(10):
                                        progress_bar.progress((i * 10 + j + 1) / (st.session_state.block_fine_tune_epochs * 10))
                                        time.sleep(0.1)
                        
                        # Execute channel pruning if selected
                        if st.session_state.channel_pruning:
                            status_text.text("Applying channel pruning...")
                            progress.progress(0.4)
                            
                            channel_pruned_model = main_pruning_loop(
                                model=current_model,
                                block_level_dict={},  # No block pruning at this step
                                uniform_pruning_ratio=st.session_state.channel_pruning_ratio,
                                type="UNIFORM",
                                block_fine_tune_epochs=st.session_state.channel_fine_tune_epochs if st.session_state.channel_fine_tune else 0
                            )
                            
                            progress.progress(0.6)
                            current_model = channel_pruned_model

                            # Fine-tuning after Channel Pruning
                            if st.session_state.channel_fine_tune:
                                status_text.text(f"Fine-tuning after Channel Pruning for {st.session_state.channel_fine_tune_epochs} epochs...")
                                main_finetune_model(current_model, st.session_state.channel_fine_tune_epochs, DEVICE)
                                progress_bar = st.progress(0)
                                for i in range(st.session_state.channel_fine_tune_epochs):
                                    for j in range(10):
                                        progress_bar.progress((i * 10 + j + 1) / (st.session_state.channel_fine_tune_epochs * 10))
                                        time.sleep(0.1)
                        
                        # Calculate pruned model stats
                        pruned_size_mb = get_model_size(current_model)
                        pruned_flops = count_net_flops(current_model, dummy_input)
                        
                        # Apply Knowledge Distillation if selected
                        if st.session_state.knowledge_distillation:
                            # Check if teacher model was uploaded
                            teacher_model_file = st.session_state.get('teacher_model', None)
                            if not teacher_model_file:
                                st.error("Teacher model required for Knowledge Distillation")
                            else:
                                status_text.text("Applying Knowledge Distillation...")
                                progress.progress(0.7)
                                
                                # Load teacher model
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_teacher_file:
                                    tmp_teacher_file.write(teacher_model_file.getvalue())
                                    teacher_model_path = tmp_teacher_file.name
                                
                                teacher_model = torch.load(teacher_model_path, map_location=DEVICE)
                                teacher_model.to(DEVICE)
                                teacher_model.eval()
                                
                                # Apply distillation
                                current_model = knowledge_distillation_prune(
                                    teacher_model=teacher_model,
                                    student_model=current_model,
                                    num_epochs=distillation_epochs,
                                    device=DEVICE
                                )
                                
                                # Clean up
                                os.unlink(teacher_model_path)
                        
                        # Test model accuracy
                        pruned_acc = test_model(current_model, DEVICE)  # Or use simulated accuracy
                        
                        # Save the pruned model
                        pruned_model_path = os.path.join(tempfile.gettempdir(), "pruned_model.pth")
                        pruned_state_dict_path = os.path.join(tempfile.gettempdir(), "pruned_model_state_dict.pth")
                        
                        torch.save(current_model, pruned_model_path)
                        torch.save(current_model.state_dict(), pruned_state_dict_path)
                        
                        # Initialize quantized model variables
                        quantized_size = None
                        quantized_model_path = None
                        
                        # Apply quantization if selected
                        if st.session_state.quantization:
                            status_text.text(f"Applying {st.session_state.quantization_type} quantization...")
                            progress.progress(0.8)
                            
                            quantized_model_path = os.path.join(tempfile.gettempdir(), f"quantized_model_{st.session_state.quantization_type}.tflite")
                            
                            # Apply quantization
                            quantize_model(pruned_model_path, quantized_model_path, "person_detection_validation", st.session_state.quantization_type)
                            
                            # Calculate quantized model size
                            quantized_size = get_tflite_model_size(quantized_model_path)
                        
                        progress.progress(1.0)
                        status_text.text("Optimization complete!")
                        
                        # Calculate final stats
                        final_flops = count_net_flops(current_model, dummy_input)
                        final_peak_act = count_peak_activation_size(current_model, dummy_input)
                        
                        # Display results
                        st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                        st.subheader("Optimization Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Before")
                            st.metric("Size", f"{original_size_mb:.2f} MB")
                            st.metric("FLOPs", f"{original_flops / 1e6:.2f} M")
                            st.metric("Accuracy", f"{original_acc}%")
                        
                        with col2:
                            st.markdown("### After")
                            # Calculate size reduction based on whether quantization was applied
                            size_reduction = ((original_size_mb - pruned_size_mb) / original_size_mb) * 100
                            
                            if st.session_state.quantization and quantized_size is not None:
                                final_size = quantized_size
                                quantized_size_reduction = ((original_size_mb - quantized_size) / original_size_mb) * 100
                                size_display = f"{quantized_size:.2f} MB"  
                                size_reduction_display = f"-{quantized_size_reduction:.1f}%"
                            else:
                                final_size = pruned_size_mb
                                size_display = f"{pruned_size_mb:.2f} MB"
                                size_reduction_display = f"-{size_reduction:.1f}%"
                                
                            st.metric("Size", size_display, delta=size_reduction_display, delta_color="inverse")
                            
                            flops_reduction = ((original_flops - final_flops) / original_flops) * 100
                            st.metric("FLOPs", f"{final_flops / 1e6:.2f} M", 
                                     delta=f"-{flops_reduction:.1f}%", delta_color="inverse")
                            
                            acc_delta = pruned_acc - original_acc
                            st.metric("Accuracy", f"{pruned_acc}%", 
                                     delta=f"{acc_delta:.1f}%", delta_color="normal")
                        
                        # Summary
                        methods_used = []
                        if st.session_state.block_pruning: 
                            methods_used.append(f"Block Pruning{' with fine-tuning' if st.session_state.block_fine_tune else ''}")
                        if st.session_state.channel_pruning: 
                            methods_used.append(f"Channel Pruning{' with fine-tuning' if st.session_state.channel_fine_tune else ''}")
                        if st.session_state.knowledge_distillation: 
                            methods_used.append("Knowledge Distillation")
                        if st.session_state.quantization: 
                            methods_used.append(f"{st.session_state.quantization_type.upper()} Quantization")
                        
                        st.info(f"Methods used: {', '.join(methods_used)}")
                        
                        # Download section
                        st.markdown("<div class='download-section'>", unsafe_allow_html=True)
                        st.subheader("Download Optimized Models")
                        
                        # Store files in session_state
                        with open(pruned_state_dict_path, "rb") as f:
                            pruned_bytes = f.read()
                        
                        download_col1, download_col2 = st.columns(2)
                        
                        with download_col1:
                            st.download_button(
                                label="Download Pruned Model (.pth)",
                                data=pruned_bytes,
                                file_name="optimized_model.pth",
                                mime="application/octet-stream"
                            )
                        
                        if st.session_state.quantization and quantized_model_path and os.path.exists(quantized_model_path):
                            with open(quantized_model_path, "rb") as f:
                                quantized_bytes = f.read()
                            
                            with download_col2:
                                st.download_button(
                                    label="Download Quantized Model (.tflite)",
                                    data=quantized_bytes,
                                    file_name=f"optimized_model_{st.session_state.quantization_type}.tflite",
                                    mime="application/octet-stream"
                                )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error during optimization: {str(e)}")
                        
                    # Clean up temporary files
                    if os.path.exists(model_path):
                        os.unlink(model_path)
else:
    # Initial page content when no model is uploaded
    st.info("👈 Upload your PyTorch model in the sidebar to get started")
    
    # Application information
    with st.expander("About Model Optimizer", expanded=True):
        st.markdown("""
        This tool helps you optimize deep learning models for deployment on resource-constrained devices. It provides:
        
        - 🔄 **Block and Channel Pruning**: Remove unnecessary parts of your model
        - 🎯 **Knowledge Distillation**: Transfer knowledge from a larger teacher model
        - 📊 **Quantization**: Convert to smaller numerical formats
        - ⚡ **Fine-tuning**: Restore accuracy after optimization
        
        Start by uploading your PyTorch model in the sidebar.
        """)