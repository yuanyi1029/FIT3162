import streamlit as st
import torch
import os
import tempfile
import torch.nn as nn
import random

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
    
    # Pruning parameters
    st.subheader("Pruning Parameters")
    pruning_ratio = st.slider("Pruning Ratio", 0.0, 0.9, 0.5, 0.05)
    
    # Additional parameters
    pruning_method = st.selectbox("Pruning Method", ["L1 Norm", "L2 Norm", "Random"])
    fine_tune = st.checkbox("Fine-tune after pruning")
    fine_tune_epochs = 0
    if fine_tune:
        fine_tune_epochs = st.number_input("Fine-tuning epochs", 1, 100, 5)
    
    # Prune button
    if st.button("Prune Model"):
        with st.spinner("Pruning model..."):
            try:
                # Call your pruning function
                # pruned_model, stats = prune_model(
                #     model_path=model_path,
                #     pruning_ratio=pruning_ratio,
                #     method=pruning_method.split()[0].lower(),
                #     fine_tune=fine_tune,
                #     fine_tune_epochs=fine_tune_epochs
                # )
                
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
