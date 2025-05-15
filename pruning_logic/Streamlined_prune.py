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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import torch.optim as optim
import pandas as pd 
from torch import nn
from mcunet.tinynas.search.accuracy_predictor import (
    AccuracyDataset,
    MCUNetArchEncoder,
)

from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.utils.mcunet_eval_helper import calib_bn, validate
from mcunet.utils.arch_visualization_helper import draw_arch
from mcunet.utils.pytorch_utils import count_peak_activation_size, count_net_flops, count_parameters
from mcunet.utils.pytorch_modules import SEModule as SEClass 

from mcunet.tinynas.nn.networks.mcunets import MobileInvertedResidualBlock


from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#For Nvidia GPU 
#device = 'cuda:01'

#for m1 macbook
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

#import from Pruning module 
from .Pruning_definitions import (
    analyze_model,  
    evaluate, 
    build_val_data_loader, 
    get_model_size,
    plot_weight_distribution,
    sensitivity_scan,
    plot_sensitivity_scan,
    apply_conv2d_sorting,
    uniform_channel_prune,
    col_based_prune_reduction,
    Pruner,
    AnalyticalEfficiencyPredictor,
    test_model_finetune,
    freeze_layers,
    prune_mb_inverted_block,
    finetune_model
)

#Prepare Grayscale Data Loading


def prepare_dataloaders(data_dir, image_size=96, batch_size=32, seed=42):
    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # --- Datasets ---
    full_dataset_train = datasets.ImageFolder(data_dir, transform=train_transform)
    full_dataset_eval = datasets.ImageFolder(data_dir, transform=eval_transform)

    total_len = len(full_dataset_train)
    train_size = int(0.7 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(range(total_len), [train_size, val_size, test_size], generator=generator)

    train_data = Subset(full_dataset_train, train_idx)
    val_data = Subset(full_dataset_eval, val_idx)
    test_data = Subset(full_dataset_eval, test_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# Create data loaders
train_loader, val_loader, test_loader = prepare_dataloaders("person_detection_validation")


def prune_multiple_blocks(model, target_blocks_dict, fine_tune_epochs=0):
    pruned_block_model = copy.deepcopy(model)
 
    block_class = MobileInvertedResidualBlock #residual block class 
    se_class = SEClass     
    
    for target_path, pruning_ratio in target_blocks_dict.items():
        print(f"Pruning Block: {target_path}")

        #Prune
        pruned_block_model = prune_mb_inverted_block(
            model=pruned_block_model,
            target_conv_path=target_path,
            prune_ratio=pruning_ratio,
            MobileInvertedBlockClass=block_class,
            SqueezeExcitationModuleType=se_class,
            verbose=True
        )

        #Finetune only if block was pruned (ratio>0) and fine_tune_epochs>0
        if fine_tune_epochs > 0 and pruning_ratio>0:
            pruned_block_model = finetune_model(
                model=pruned_block_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs = fine_tune_epochs
            )
        

    return pruned_block_model



#Sort model channels 

def uniform_prune_and_depthwise_collapse(model, ratio): 
    model_to_prune = copy.deepcopy(model)
    
    model_to_prune = uniform_channel_prune(
        model=model_to_prune,
        prune_ratio=ratio,
        SqueezeExcitationModuleType=SEClass, 
        verbose=True
    )
    
    #Need to prune the final classifier layer to align with the number of output channels of the last conv layer 
    LAYER_TO_PRUNE = ['classifier.linear.weight']
    sparsity_config_conv = {
            'classifier.linear.weight': ratio, 
        }
    
    conv_pruner = Pruner(model_to_prune, sparsity_config_conv, mode='col-based')
    conv_pruner.apply(model_to_prune)
    
    model_to_prune = col_based_prune_reduction(model_to_prune, LAYER_TO_PRUNE)
    return model_to_prune 



def main_pruning_loop(model, block_level_dict, uniform_pruning_ratio, fine_tune_epochs, type):
    
    pruned_model = model
    
    #Block level pruning parameters 
    if type in ["BOTH", "BLOCK"]: 
        pruned_model = prune_multiple_blocks(pruned_model, block_level_dict, fine_tune_epochs)

    if type in ["BOTH", "UNIFORM"]:
        pruned_model = uniform_prune_and_depthwise_collapse(pruned_model, uniform_pruning_ratio)

    return pruned_model

#EXAMPLE OF HOW ITS USED 

