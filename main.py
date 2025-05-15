import streamlit as st
import os
import tempfile
import argparse
import json
from PIL import Image
from tqdm import tqdm
import copy
import math
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms
import torch.optim as optim 
from torch import nn
from torchvision import datasets, transforms
from mcunet.tinynas.search.accuracy_predictor import (
    AccuracyDataset,
    MCUNetArchEncoder,
)
import time
from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.utils.mcunet_eval_helper import calib_bn, validate
from mcunet.utils.pytorch_utils import count_peak_activation_size, count_net_flops, count_parameters
from mcunet.tinynas.nn.networks.mcunets import MobileInvertedResidualBlock
from pruning_logic.Streamlined_prune import main_pruning_loop 

from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import math

from pruning_logic.Pruning_definitions import get_model_size

def identify_model_blocks(model):
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
        print(f"Error while extracting block names: {e}")
        return []

st.title("Model Pruning Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your PyTorch model (.pth)", type=["pth"])

if uploaded_file:
    # Display model info
    st.success(f"Model uploaded: {uploaded_file.name}")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        model_path = tmp_file.name
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Pruning Methods Section - Changed to checkboxes
    st.subheader("Pruning Method")
    
    # Using checkboxes instead of radio buttons
    block_pruning = st.checkbox("Block Level Pruning")
    channel_pruning = st.checkbox("Uniform Channel Pruning and Depth-wise Channel Pruning")
    
    # Ensure at least one method is selected
    if not block_pruning and not channel_pruning:
        st.warning("Please select at least one pruning method.")
    else:
        # Block Level Pruning Parameters
        block_pruning_ratios = {}
        if block_pruning:
            st.write("### Block Level Pruning Parameters")
            
            # Identify model blocks
            blocks = identify_model_blocks(model)
            
            # Create a container for block-specific sliders
            blocks_container = st.container()
            
            # Create sliders for each block
            with blocks_container:
                # Option to set all blocks to the same ratio
                use_same_ratio = st.checkbox("Use same pruning ratio for all blocks", value=True)
                
                if use_same_ratio:
                    global_ratio = st.slider("Global Pruning Ratio", 0.0, 0.9, 0.5, 0.01)
                    for block in blocks:
                        block_pruning_ratios[block] = global_ratio
                else:
                    # Create two columns for the sliders to save space
                    slider_cols = st.columns(2)
                    
                    for i, block in enumerate(blocks):
                        col_idx = i % 2
                        with slider_cols[col_idx]:
                            block_pruning_ratios[block] = st.slider(
                                f"Pruning Ratio - {block}", 
                                0.0, 0.9, 0.5, 0.01,
                                key=f"block_{block}"
                            )
        
        # Uniform Channel Pruning Parameters
        channel_pruning_ratio = 0.0
        if channel_pruning:
            st.write("### Uniform Channel Pruning and Depth-wise Channel Pruning Parameters")
            channel_pruning_ratio = st.slider("Channel Pruning Ratio", 0.0, 0.9, 0.5, 0.01)
        
        # Additional parameters

        
        st.subheader("Additional Parameters")
        fine_tune = st.checkbox("Fine-tune after pruning")
        #parameter for block-level finetuning 
        fine_tune_epochs = 0
        if fine_tune:
            fine_tune_epochs = st.number_input("Fine-tuning epochs", 1, 100, 5)
        
        # Prune button
        if st.button("Prune Model"):
            with st.spinner("Pruning model..."):
                try:
                     # Display the selected pruning methods
                    selected_methods = []
                    pruning_type = ""
                    
                    if block_pruning and channel_pruning:
                        pruning_type = "BOTH"
                        selected_methods = ["Block Level Pruning", "Uniform Channel Pruning and Depth-wise Channel Pruning"]
                    elif block_pruning:
                        pruning_type = "BLOCK"
                        selected_methods = ["Block Level Pruning"]
                    elif channel_pruning:
                        pruning_type = "UNIFORM"
                        selected_methods = ["Uniform Channel Pruning and Depth-wise Channel Pruning"]
                    
                    st.write(f"Pruning with: {', '.join(selected_methods)}")
                    
                    # Calculate original model stats
                    dummy_input = (1, 1, 96, 96)

                    original_params = count_parameters(model)
                    original_size_mb = get_model_size(model)  # in mb 
                    original_flops = count_net_flops(model, dummy_input)
                    original_peak_act = count_peak_activation_size(model, dummy_input)

                    # Execute the pruning function with appropriate parameters
                    pruned_model = main_pruning_loop(
                        model=model, 
                        block_level_dict=block_pruning_ratios, 
                        uniform_pruning_ratio=channel_pruning_ratio,
                        type=pruning_type,
                        fine_tune_epochs=fine_tune_epochs
                    )
                    
                    # Calculate pruned model stats
                    pruned_params = count_parameters(pruned_model)
                    pruned_size_mb = get_model_size(pruned_model)
                    pruned_flops = count_net_flops(pruned_model, dummy_input)
                    pruned_peak_act = count_peak_activation_size(pruned_model, dummy_input)



                    size_reduction_percent = ((original_size_mb - pruned_size_mb) / original_size_mb) * 100
                    
                    # Fine-tuning if selected
                    if fine_tune:
                        st.write(f"Fine-tuning for {fine_tune_epochs} epochs...")
                        # Add fine-tuning code here if needed
                        # For demonstration, we'll just show a progress bar
                        progress_bar = st.progress(0)
                        for i in range(fine_tune_epochs):
                            # Simulate fine-tuning process
                            for j in range(10):
                                progress_bar.progress((i * 10 + j + 1) / (fine_tune_epochs * 10))
                                time.sleep(0.1)

                    # Display stats
                    st.subheader("Pruning Results")
                    st.metric("Original Model Size", f"{original_size_mb:.2f} MB")
                    st.metric("Pruned Model Size", f"{pruned_size_mb:.2f} MB")
                    st.metric("Size Reduction", f"{size_reduction_percent:.1f}%")
                    
                    # If we had actual accuracy measurements, we would display them here
                    # For now, we'll use placeholder values to maintain the UI
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Parameters", f"{original_params:.2f}M")
                        st.metric("Original FLOPs", f"{original_flops / 1e6:.2f} MFLOPs")
                        st.metric("Original Peak Activation", f"{original_peak_act / 1e6:.2f} MB")
                    with col2:
                        st.metric("Pruned Parameters", f"{pruned_params:.2f}M", 
                                delta=f"{pruned_params - original_params:.2f}M")
                        st.metric("Pruned FLOPs", f"{pruned_flops / 1e6:.2f} MFLOPs", delta=f"{(pruned_flops - original_flops) / 1e6:.2f} MFLOPs")
                        st.metric("Pruned Peak Activation", f"{pruned_peak_act / 1e6:.2f} MB", 
                                delta=f"{(pruned_peak_act - original_peak_act) / 1e6:.2f} MB")
                    
                    # Download pruned model
                    pruned_model_path = os.path.join(tempfile.gettempdir(), "pruned_model.pth")
                    torch.save(pruned_model.state_dict(), pruned_model_path)
                    
                    with open(pruned_model_path, "rb") as f:
                        st.download_button(
                            label="Download Pruned Model",
                            data=f,
                            file_name="pruned_model.pth",
                            mime="application/octet-stream"
                        )
                        
                except Exception as e:
                    st.error(f"Error during pruning: {str(e)}")
                
                # Clean up
                os.unlink(model_path)