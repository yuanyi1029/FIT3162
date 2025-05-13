import streamlit as st
import torch
import os
import tempfile
import torch.nn as nn
import random

# Add this at the top to avoid the file watcher error
import streamlit.web.bootstrap
streamlit.web.bootstrap.load_config_options(flag_options={"server.fileWatcherType": "none"})

# Function to identify blocks in a model - modified to handle the error
def identify_model_blocks(model_path):
    try:
        # Load the model with weights_only=False to handle the error
        # Note: Only do this with trusted model files
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # In a real implementation, you'd analyze the model structure
        # For now, let's return mock block names
        return [
            "layer1.0", "layer1.1", "layer1.2",
            "layer2.0", "layer2.1", "layer2.2",
            "layer3.0", "layer3.1", "layer3.2",
        ]
    except Exception as e:
        # If there's an error, return mock blocks anyway to prevent UI failure
        st.warning(e)
        st.warning(f"Warning - couldn't analyze model structure: {str(e)}")
        st.info("Using default block structure for demonstration")
        return [
            "layer1.0", "layer1.1", "layer1.2",
            "layer2.0", "layer2.1", "layer2.2",
            "layer3.0", "layer3.1", "layer3.2",
        ]

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
    
    # Pruning Methods Section - Changed to radio buttons
    st.subheader("Pruning Method")
    
    pruning_method = st.radio(
        "Select pruning method",
        ["Block Level Pruning", "Uniform Channel Pruning and Depth-wise Channel Pruning"]
    )
    
    # Block Level Pruning Parameters
    if pruning_method == "Block Level Pruning":
        st.write("### Block Level Pruning Parameters")
        
        # Identify model blocks
        blocks = identify_model_blocks(model_path)
        
        # Create a container for block-specific sliders
        blocks_container = st.container()
        
        # Create sliders for each block
        block_pruning_ratios = {}
        
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
    elif pruning_method == "Uniform Channel Pruning and Depth-wise Channel Pruning":
        st.write("### UUniform Channel Pruning and Depth-wise Channel Pruning Parameters")
        channel_pruning_ratio = st.slider("Channel Pruning Ratio", 0.0, 0.9, 0.5, 0.01)
    
    # Additional parameters
    st.subheader("Additional Parameters")
    fine_tune = st.checkbox("Fine-tune after pruning")
    fine_tune_epochs = 0
    if fine_tune:
        fine_tune_epochs = st.number_input("Fine-tuning epochs", 1, 100, 5)
    
    # Prune button
    if st.button("Prune Model"):
        with st.spinner("Pruning model..."):
            try:
                # Display the selected pruning method
                st.write(f"Pruning with: {pruning_method}")
                
                if pruning_method == "Block Level Pruning":
                    # Pass block_pruning_ratios to your function
                    st.write("Block ratios selected:", block_pruning_ratios)
                else:
                    # Pass channel_pruning_ratio to your function
                    st.write(f"Channel pruning ratio: {channel_pruning_ratio}")
                
                # Mock model for demonstration
                pruned_model = nn.Sequential(
                    nn.Linear(10, 5),
                    nn.ReLU(),
                    nn.Linear(5, 2)
                )

                # Mocked stats
                stats = {
                    "original_size_mb": random.uniform(5.0, 10.0),
                    "pruned_size_mb": random.uniform(2.0, 4.0),
                    "size_reduction_percent": random.uniform(50.0, 70.0),
                    "accuracy_before": random.uniform(80.0, 90.0),
                    "accuracy_after": random.uniform(78.0, 88.0)
                }

                # Display stats
                st.subheader("Pruning Results")
                st.metric("Original Model Size", f"{stats['original_size_mb']:.2f} MB")
                st.metric("Pruned Model Size", f"{stats['pruned_size_mb']:.2f} MB")
                st.metric("Size Reduction", f"{stats['size_reduction_percent']:.1f}%")
                
                # Performance metrics
                if 'accuracy_before' in stats and 'accuracy_after' in stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Accuracy", f"{stats['accuracy_before']:.2f}%")
                    with col2:
                        st.metric("Pruned Accuracy", f"{stats['accuracy_after']:.2f}%", 
                                delta=f"{stats['accuracy_after'] - stats['accuracy_before']:.2f}%")
                
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