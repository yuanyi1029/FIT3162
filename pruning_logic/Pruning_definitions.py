
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

from mcunet.tinynas.elastic_nn.networks.ofa_mcunets import OFAMCUNets
from mcunet.utils.mcunet_eval_helper import calib_bn, validate
from mcunet.utils.arch_visualization_helper import draw_arch
from mcunet.utils.pytorch_utils import count_peak_activation_size, count_net_flops, count_parameters


from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import math


#For Nvidia GPU 
#device = 'cuda:01'

#for m1 macbook
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


#------ ALL HELPER FUNCTIONS --------
 
def count_parameters(model):
    """
    Count the number of parameters in a model.

    Args:
        model (nn.Module): The PyTorch model.
        only_trainable (bool): If True, count only parameters that require gradients (i.e., are trainable).

    Returns:
        int: Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def freeze_layers(model, freeze_amount):
    # Step 1: Flatten all modules in order
    all_layers = list(model.modules())

    # Step 2: Filter only layers with parameters
    trainable_layers = [layer for layer in all_layers if any(p.requires_grad for p in layer.parameters(recurse=False))]

    # Step 3: Determine split point
    total_layers = len(trainable_layers)
    num_freeze = int(freeze_amount * total_layers)

    print(f"Total trainable layers: {total_layers}")
    print(f"Freezing first {num_freeze} layers...")

    # Step 4: Freeze first
    for layer in trainable_layers[:num_freeze]:
        for param in layer.parameters():
            param.requires_grad = False



def analyze_model(model, device, input_size=(1, 3, 224, 224)):
    model = model.to(device)
    print("Model Architecture:\n", model)
    print("\n====================\n")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print("\n====================\n")

    # Register hooks to track activations
    activation_sizes = {}

    def hook_fn(module, input, output):
        activation_sizes[module] = output.shape

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d)):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Run a forward pass with a dummy input
    dummy_input = torch.randn(*input_size).to(device)
    model(dummy_input)

    # Print activation sizes
    print("Activation Sizes (Output Shapes per Layer):")
    for layer, size in activation_sizes.items():
        print(f"{layer}: {size}")

    # Remove hooks
    for hook in hooks:
        hook.remove()


def evaluate(model: nn.Module,dataloader, device, verbose=True) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
    # Move the data from CPU to GPU
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Inference
    outputs = model(inputs)

    # Convert logits to class indices
    outputs = outputs.argmax(dim=1)

    # Update metrics
    num_samples += targets.size(0) #get batch size (num of samples)

    num_correct += (outputs == targets).sum()


  return (num_correct / num_samples * 100).item()


# def evaluate_model(model, dataloader, device):
#     """
#     Evaluates the given PyTorch regression model on a dataset using multiple metrics.

#     Args:
#         model (torch.nn.Module): The PyTorch model to evaluate.
#         dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset.
#         device (str): The device to use ('cuda' or 'cpu').

#     Returns:
#         dict: A dictionary containing 'mse_loss', 'mae_loss', 'rmse_loss', and 'r2_score'.
#     """
#     model.eval()  # Set model to evaluation mode
#     total_loss = 0
#     total_mae = 0
#     correct = 0
#     total_samples = 0
#     sum_squared_errors = 0
#     sum_squared_total = 0

#     criterion = torch.nn.MSELoss()  # Mean Squared Error
#     with torch.no_grad():  # No gradients needed during evaluation
#         all_targets = []
#         all_outputs = []

#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs).squeeze()  # Remove size-1 dimensions for correct shape

#             # Compute MSE Loss
#             loss = criterion(outputs, targets)
#             total_loss += loss.item() * inputs.size(0)

#             # Compute MAE
#             mae_loss = torch.abs(outputs - targets).mean()
#             total_mae += mae_loss.item() * inputs.size(0)

#             # Store predictions and targets for R^2 calculation
#             all_targets.append(targets.cpu())
#             all_outputs.append(outputs.cpu())

#             total_samples += inputs.size(0)

#     # Convert to single tensors
#     all_targets = torch.cat(all_targets)
#     all_outputs = torch.cat(all_outputs)

#     # Compute RMSE
#     mse = total_loss / total_samples
#     rmse = mse ** 0.5
#     mae = total_mae / total_samples

#     # Compute R-squared (R²) score
#     mean_target = all_targets.mean()
#     sum_squared_errors = torch.sum((all_targets - all_outputs) ** 2)
#     sum_squared_total = torch.sum((all_targets - mean_target) ** 2)
#     r2_score = 1 - (sum_squared_errors / sum_squared_total).item()

#     return {
#         "mse_loss": mse,
#         "mae_loss": mae,
#         "rmse_loss": rmse,
#         "r2_score": r2_score
#     }


#Model finetuning function: 

def finetune_model(
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-4,
    weight_decay=3e-5,
    num_epochs=50,
    patience=5
):
    model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    no_improve_epochs = 0

    print(f"Starting fine-tuning for max {num_epochs} epochs with patience={patience}...")

    for epoch in range(num_epochs):
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total * 100

        # --- Validation ---
        # model.eval()
        # val_loss, val_correct, val_total = 0.0, 0, 0
        # with torch.no_grad():
        #     for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)

        #         val_loss += loss.item() * inputs.size(0)
        #         _, predicted = torch.max(outputs, 1)
        #         val_total += labels.size(0)
        #         val_correct += (predicted == labels).sum().item()

        # epoch_val_loss = val_loss / len(val_loader.dataset)
        # epoch_val_acc = val_correct / val_total * 100
        # current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train Loss={epoch_train_loss:.4f}, Acc={epoch_train_acc:.2f}%")
            #   f"Val Loss={epoch_val_loss:.4f}, Acc={epoch_val_acc:.2f}% | LR={current_lr:.6f}")

        # if epoch_val_acc > best_val_acc:
        #     best_val_acc = epoch_val_acc
        #     no_improve_epochs = 0
        #     print(f"  -> Validation accuracy improved to {best_val_acc:.2f}%")
        # else:
        #     no_improve_epochs += 1
        #     print(f"  -> No improvement ({no_improve_epochs} epochs)")
        #     if no_improve_epochs >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break

        scheduler.step()

    print("Fine-tuning complete.")
    return model

def test_model_finetune(model_test_finetune, test_loader):
    correct = 0
    total = 0

    model_test_finetune.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_test_finetune(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 4. Print the initial accuracy before fine-tuning
    initial_accuracy = correct / total * 100
    print(f"Accuracy on Test Set: {initial_accuracy:.2f}%")
    return initial_accuracy



    
def build_val_data_loader(data_dir, resolution, batch_size=128, split=0):
    # split = 0: real val set, split = 1: holdout validation set
    assert split in [0, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {"num_workers": min(8, os.cpu_count()), "pin_memory": False}

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution)
            ),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    val_dataset = torch.utils.data.Subset(
        val_dataset, list(range(len(val_dataset)))[split::2]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return val_loader


def get_model_size(model):
    """
    Computes the size of a PyTorch model in megabytes (MB).
    
    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
    
    Returns:
        float: Model size in MB.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) # Computes the number of elements and their byte size for each parameter.
    print("Number of params: ", param_size)
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  #Computes the same for registered buffers (e.g., BatchNorm running stats).
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert bytes to MB
    print("Model size in MB: ", total_size)
    return total_size



def _get_module_by_name(model: nn.Module, name: str):
    """Gets a module submodule using its dot-separated name."""
    names = name.split('.')
    module = model
    for n in names:
        # Handle cases where module is a Sequential or ModuleList
        if n.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
             module = module[int(n)]
        else:
            try:
                module = getattr(module, n)
            except AttributeError:
                 raise AttributeError(f"Module {model.__class__.__name__} has no attribute {n} in name {name}")
    return module



def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Sets a module submodule using its dot-separated name."""
    names = name.split('.')
    parent_module = model
    for i, n in enumerate(names[:-1]):
         if n.isdigit() and isinstance(parent_module, (nn.Sequential, nn.ModuleList)):
             parent_module = parent_module[int(n)]
         else:
            try:
                parent_module = getattr(parent_module, n)
            except AttributeError:
                raise AttributeError(f"Module {model.__class__.__name__} has no parent for attribute {n} in name {name}")

    final_name = names[-1]
    if final_name.isdigit() and isinstance(parent_module, (nn.Sequential, nn.ModuleList)):
        parent_module[int(final_name)] = new_module
    else:
        setattr(parent_module, final_name, new_module)



#------ END OF HELPER FUNCTIONS --------





# ----- ALL ANALYSIS FUNCTIONS ------


class AnalyticalEfficiencyPredictor:
    def __init__(self):
        pass


    def get_efficiency(self, subnet):
        if torch.cuda.is_available():
            subnet = subnet.cuda()
            
        # The data shape is (batch_size, input_channel, image_size, image_size)
        batch_size = 1 #  efficiency is measured for processing a single image at a time.
        input_channel = 1
        image_size = 96
        data_shape = (batch_size, input_channel, image_size, image_size)

        macs = count_net_flops(subnet, data_shape)  # Compute MACs
        peak_memory = count_peak_activation_size(subnet, data_shape)  # Compute peak memory in bytes

        #add total memory of model 
        model_size = get_model_size(subnet)
        
        
        return dict(millionMACs=macs / 1e6, KBPeakMemory=peak_memory / 1024, modelSizeMB=model_size)

    
    def satisfy_constraint(self, measured: dict, target: dict):
        for key in measured:
            # if the constraint is not specified, we just continue
            if key not in target:
                continue
            # if we exceed the constraint, just return false.
            if measured[key] > target[key]:
                return False
        # no constraint violated, return true.
        return True
    

@torch.no_grad()
def sensitivity_scan(model, dataloader, device, layers_to_scan = None, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) in model.named_parameters() if param.dim() > 1 and name in layers_to_scan]

    # counter = 0
    # limit = 5 
    
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, device, verbose=False)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\r sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]', end='')
        accuracies.append(accuracy)
        
        # counter += 1 
        # if limit <= counter:
        #     break 
    return sparsities, accuracies


def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy, model, layers_to_scan):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    

    
    # Ensure we have matching accuracy lists for selected layers
    accuracies = accuracies[:len(layers_to_scan)]

    # Determine subplot grid layout
    num_layers = len(layers_to_scan)
    num_cols = math.ceil(math.sqrt(num_layers))  # Square-like layout
    num_rows = math.ceil(num_layers / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    axes = np.ravel(axes)  # Flatten into a list

    for plot_index, (name, acc) in enumerate(zip(layers_to_scan, accuracies)):
        ax = axes[plot_index]
        ax.plot(sparsities, acc, label='Accuracy after pruning')
        ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities), linestyle='dashed', label=f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy')

        ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
        ax.set_ylim(50, 95)
        ax.set_title(name)  # Display layer name as title
        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Top-1 Accuracy')
        ax.legend()
        ax.grid(axis='x')

    # Hide unused subplots if fewer layers are selected than grid size
    for i in range(len(layers_to_scan), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


def plot_weight_distribution(model, layers_to_scan, bins=256, count_nonzero_only=False):
    # Filter model parameters to only include layers in layers_to_scan
    selected_layers = [(name, param) for name, param in model.named_parameters() if name in layers_to_scan]

    num_layers = len(selected_layers)
    num_cols = math.ceil(math.sqrt(num_layers))  # Square-like layout
    num_rows = math.ceil(num_layers / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(min(5 * num_cols, 50), min(5 * num_rows, 50)))
    axes = np.ravel(axes)  # Flatten into a list

    for plot_index, (name, param) in enumerate(selected_layers):
        ax = axes[plot_index]
        
        param_cpu = param.detach().view(-1).cpu()
        if count_nonzero_only:
            param_cpu = param_cpu[param_cpu != 0]  # Remove zeros
        
        ax.hist(param_cpu, bins=bins, density=True, color='blue', alpha=0.5)
        #density = count in that bin / total number of weights * bin width 
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f"Weight Distribution: {name}")

    # Hide unused subplots if fewer layers are selected
    for i in range(len(selected_layers), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


def plot_num_parameters_distribution(model, fixed_ymax = None):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()

    
        
    fig = plt.figure(figsize=(100, 60))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))

    if fixed_ymax is not None:
        plt.ylim(top=fixed_ymax)
        
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

 


# ------ END OF ANALYSIS FUNCTIONS ------ 



# ------ ALL FC PRUNING FUNCTIONS ------ 

class Pruner:
    def __init__(self, model, sparsity_dict, mode):
      #self.masks is a dictionary of key = layer name, value = mask (1/0 matrix)
        self.masks = Pruner.prune(model, sparsity_dict, mode)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict, mode:str):
        masks = dict()
        for name, param in model.named_parameters():
            if name in sparsity_dict and param.dim() > 1: # we only prune conv and fc weights
                if mode == 'fine-grained':
                    masks[name] = fine_grained_prune(param, sparsity_dict[name])
                if mode == 'row-based':
                    masks[name] = row_based_prune(param, sparsity_dict[name])
                if mode == 'col-based':
                    masks[name] = column_based_prune(param, sparsity_dict[name])
        return masks
    


def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """

    #ensure sparsity within 0 and 1
    sparsity = min(max(0.0, sparsity), 1.0)


    if sparsity == 1.0:
          # If sparsity is 1.0, prune everything (set all weights to zero)
        tensor.zero_()
        return torch.zeros_like(tensor)

    elif sparsity == 0.0:
        #If sparsity is 0, no pruning so everything int the tensor is kept at 1
        return torch.ones_like(tensor)

    num_elements = tensor.numel()


    # Step 1: calculate the #zeros after pruning (please use round())
    #we need to calculate how many weights should be set to zero
    num_zeros = round(num_elements * sparsity)

    # Step 2: calculate the importance of weight tensor
    importance = torch.abs(tensor)

    # Step 3: calculate the pruning threshold
    # Find the threshold value using kthvalue (prune the smallest |W| values)
#The first value (values) is the k-th smallest element itself.
#The second value (indices) gives the position of the k-th smallest element in the original flattened tensor.
    # threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]

    threshold = torch.topk(importance.flatten(), k=num_zeros, largest=False)[0][-1] #idx 0 is the value tensor, idx 1 is the tensor index 


    # Step 4: get binary mask (1 for nonzeros, 0 for zeros)
    mask = importance > threshold


    # Step 5: apply mask to prune the tensor
    tensor.mul_(mask)

    return mask




def row_based_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Row-based pruning using L1 norm, setting entire rows to zero.
    
    :param tensor: torch.Tensor, weight matrix of a layer
    :param sparsity: float, pruning sparsity (fraction of rows to prune)
    :return: torch.Tensor, binary mask (1 = keep, 0 = prune)
    """

    # Ensure sparsity is between 0 and 1
    sparsity = min(max(0.0, sparsity), 1.0)
    
    num_rows = tensor.shape[0]
    num_prune = round(num_rows * sparsity)  # Number of rows to prune

    if num_prune == 0:
        return torch.ones_like(tensor)  # No pruning

    if num_prune == num_rows:
        return torch.zeros_like(tensor)  # Prune everything

    # Compute L1 norm for each row
    row_norms = torch.norm(tensor, p=1, dim=1)  # L1 norm along rows

    # Find threshold: top-k lowest L1 norms are pruned
    threshold = torch.topk(row_norms, k=num_prune, largest=False)[0][-1]

    # Create binary mask: 1 = keep, 0 = prune entire row
    row_mask = (row_norms > threshold).float().unsqueeze(1)  # Shape: [num_rows, 1]
    mask = row_mask * torch.ones_like(tensor)  # Broadcast mask to full shape

    # Apply mask
    #not modifying tensor in place 
    # tensor.mul_(mask)

    return mask



@torch.no_grad()
def row_based_prune_reduction(model: nn.Module, layers_to_reduce: list[str]) -> nn.Module:
    """
    Reduces the dimensions of specified layers after row-based pruning
    by removing zeroed rows and adjusting subsequent layers accordingly.

    Assumes the model has already been pruned using a Pruner (so rows are zeroed).
    Modifies the model in place.

    Args:
        model (nn.Module): The PyTorch model (already row-based pruned).
        layers_to_reduce (list[str]): List of layer names (weights) whose rows
                                      should be physically removed.
                                      Example: ['fc1.weight', 'conv2.weight'].

    Returns:
        nn.Module: The modified model with reduced layers.

    Raises:
        ValueError: If a layer type is unsupported or dimensions mismatch.
        AttributeError: If a layer name is not found.
        NotImplementedError: If automatic subsequent layer adjustment fails for a structure.
    """
    model_device = next(model.parameters()).device # Get model device
    
    # Use OrderedDict to preserve order, important for finding subsequent layers
    module_list = list(model.named_modules()) 
    module_dict = OrderedDict(module_list)
    module_names = list(module_dict.keys())

    # Store mapping from pruned layer name to indices kept and new output size
    reduction_info = {}

    print("--- Starting Row Pruning Reduction ---")

    # --- Pass 1: Reduce rows in specified layers ---
    for layer_weight_name in layers_to_reduce:
        if not layer_weight_name.endswith('.weight'):
            print(f"Warning: Skipping '{layer_weight_name}'. Expecting layer weight name.")
            continue
            
        layer_name = layer_weight_name.replace('.weight', '')
        print(f"Processing layer '{layer_name}' for row reduction...")

        try:
            layer_module = _get_module_by_name(model, layer_name)
            layer_weight = layer_module.weight.data # Get the current weight tensor
            has_bias = hasattr(layer_module, 'bias') and layer_module.bias is not None
            if has_bias:
                layer_bias = layer_module.bias.data # Get the current bias tensor
            else:
                layer_bias = None
                
        except AttributeError:
            print(f"Error: Layer '{layer_name}' not found in the model.")
            continue

        # Identify non-zero rows (rows to keep)
        # Use a small tolerance for floating point comparisons
        # Sum absolute values across all dimensions except the first (row dim)
        row_sums = torch.sum(torch.abs(layer_weight), dim=list(range(1, layer_weight.dim())))
        keep_indices = torch.where(row_sums > 1e-8)[0] # Indices of rows to keep
        
        original_rows = layer_weight.shape[0]
        new_rows = len(keep_indices)

        if new_rows == original_rows:
            print(f"Layer '{layer_name}' has no fully zeroed rows to remove. Skipping reduction.")
            continue
            
        if new_rows == 0:
            print(f"Warning: All rows in layer '{layer_name}' are zero. This layer will be removed or output zero.")

        print(f"  Reducing layer '{layer_name}': {original_rows} rows -> {new_rows} rows.")

        # Create new weight and bias tensors
        new_weight = layer_weight[keep_indices, ...].clone() # Ellipsis handles Conv layers correctly
        new_bias = layer_bias[keep_indices].clone() if has_bias else None

        # Create the new layer with reduced output dimension
        new_layer = None
        if isinstance(layer_module, nn.Linear):
            new_layer = nn.Linear(in_features=layer_module.in_features,
                                  out_features=new_rows,
                                  bias=has_bias,
                                  device=model_device)
            
        elif isinstance(layer_module, nn.Conv2d):
             new_layer = nn.Conv2d(in_channels=layer_module.in_channels,
                                   out_channels=new_rows,
                                   kernel_size=layer_module.kernel_size,
                                   stride=layer_module.stride,
                                   padding=layer_module.padding,
                                   dilation=layer_module.dilation,
                                   groups=layer_module.groups,
                                   bias=has_bias,
                                   padding_mode=layer_module.padding_mode,
                                   device=model_device)
            
        # Add other layer types (Conv1d, Conv3d, etc.) if needed
        else:
            print(f"Warning: Layer type {type(layer_module)} not currently supported for reduction. Skipping '{layer_name}'.")
            continue

        # Assign the new weights and bias
        new_layer.weight.data = new_weight
        if has_bias:
            new_layer.bias.data = new_bias

        # Replace the old layer with the new one in the model
        _set_module_by_name(model, layer_name, new_layer)
        
        # Store info needed for adjusting the *next* layer
        reduction_info[layer_name] = {
            'keep_indices': keep_indices,
            'new_output_dim': new_rows
        }

    # --- Pass 2: Adjust subsequent layers ---
    print("\n--- Adjusting Subsequent Layers ---")
    adjusted_layers = set() # Keep track of layers already adjusted

    # Iterate through the ordered list of modules again
    for i, (current_layer_name, current_module) in enumerate(module_list):
        if not current_layer_name or current_layer_name in adjusted_layers: # Skip empty names or already processed layers
            continue

        # Check if the *previous* layer in the sequence was reduced
        # This relies on the order from model.named_modules() being sequential
        previous_layer_name = None
        if i > 0:
             # Find the *actual* module name that might have been reduced before this one
             # Walk backwards skipping non-parameter layers if necessary (e.g., activations, dropout)
             for j in range(i - 1, -1, -1):
                 potential_prev_name, potential_prev_module = module_list[j]
                 if isinstance(potential_prev_module, (nn.Linear, nn.Conv2d)): # Check if it's a layer type we handle
                     if potential_prev_name in reduction_info:
                        previous_layer_name = potential_prev_name
                        break # Found the relevant previous layer that was reduced
                     else:
                         # Found a parameter layer, but it wasn't reduced, so stop looking back for this current_module
                         break 
             
        if previous_layer_name and previous_layer_name in reduction_info:
            print(f"Adjusting layer '{current_layer_name}' based on reduction of '{previous_layer_name}'...")
            
            info = reduction_info[previous_layer_name]
            keep_indices = info['keep_indices']
            expected_input_dim = info['new_output_dim']

            # Get the current layer that needs input adjustment
            try:
                 layer_to_adjust = _get_module_by_name(model, current_layer_name) # Get potentially updated module reference
                 layer_weight = layer_to_adjust.weight.data
                 has_bias = hasattr(layer_to_adjust, 'bias') and layer_to_adjust.bias is not None
                 layer_bias = layer_to_adjust.bias.data if has_bias else None # Bias is usually unaffected by input dim change
            except AttributeError:
                 print(f"Error: Could not find layer '{current_layer_name}' to adjust.")
                 continue # Should not happen if iteration is correct

            new_layer = None
            if isinstance(layer_to_adjust, nn.Linear):
                original_input_dim = layer_to_adjust.in_features
                if original_input_dim != len(keep_indices):
                    # Check if already adjusted indirectly? This shouldn't happen with the adjusted_layers set
                    if current_layer_name in adjusted_layers: continue 
                    
                    print(f"  Adjusting Linear layer '{current_layer_name}': in_features {original_input_dim} -> {expected_input_dim}")
                    
                    # Select columns corresponding to the kept indices of the previous layer
                    new_weight = layer_weight[:, keep_indices].clone()
                    
                    new_layer = nn.Linear(in_features=expected_input_dim,
                                          out_features=layer_to_adjust.out_features,
                                          bias=has_bias,
                                          device=model_device)
                    new_layer.weight.data = new_weight
                    if has_bias:
                         new_layer.bias.data = layer_bias # Bias remains the same

            elif isinstance(layer_to_adjust, nn.Conv2d):
                original_input_channels = layer_to_adjust.in_channels
                 # Check if groups might affect which indices to keep (more complex case)
                if layer_to_adjust.groups != 1 and layer_to_adjust.groups != original_input_channels:
                     print(f"Warning: Grouped convolution found in '{current_layer_name}'. Automatic input channel reduction might be incorrect. Skipping adjustment.")
                     continue # Skip grouped conv unless groups==in_channels (depthwise)
                
                # Check if this layer's input dim already matches
                if original_input_channels != expected_input_dim:
                    # Check if already adjusted indirectly?
                    if current_layer_name in adjusted_layers: continue
                    
                    print(f"  Adjusting Conv2d layer '{current_layer_name}': in_channels {original_input_channels} -> {expected_input_dim}")
                    
                    # Select input channels (dim 1) corresponding to the kept indices
                    new_weight = layer_weight[:, keep_indices, :, :].clone()
                    
                    new_layer = nn.Conv2d(in_channels=expected_input_dim,
                                          out_channels=layer_to_adjust.out_channels,
                                          kernel_size=layer_to_adjust.kernel_size,
                                          stride=layer_to_adjust.stride,
                                          padding=layer_to_adjust.padding,
                                          dilation=layer_to_adjust.dilation,
                                          groups=layer_to_adjust.groups if layer_to_adjust.groups==1 else expected_input_dim, # Adjust groups for depthwise
                                          bias=has_bias,
                                          padding_mode=layer_to_adjust.padding_mode,
                                          device=model_device)
                                          
                    new_layer.weight.data = new_weight
                    if has_bias:
                         new_layer.bias.data = layer_bias # Bias remains the same
            
            elif isinstance(layer_to_adjust, (nn.BatchNorm1d, nn.BatchNorm2d)):
                 # Adjust BatchNorm layers
                 original_num_features = layer_to_adjust.num_features
                 if original_num_features != expected_input_dim:
                     if current_layer_name in adjusted_layers: continue
                     
                     print(f"  Adjusting BatchNorm layer '{current_layer_name}': num_features {original_num_features} -> {expected_input_dim}")
                     
                     # Create new BatchNorm layer
                     if isinstance(layer_to_adjust, nn.BatchNorm1d):
                         new_layer = nn.BatchNorm1d(num_features=expected_input_dim,
                                                    eps=layer_to_adjust.eps,
                                                    momentum=layer_to_adjust.momentum,
                                                    affine=layer_to_adjust.affine,
                                                    track_running_stats=layer_to_adjust.track_running_stats,
                                                    device=model_device)
                     else: # BatchNorm2d
                          new_layer = nn.BatchNorm2d(num_features=expected_input_dim,
                                                     eps=layer_to_adjust.eps,
                                                     momentum=layer_to_adjust.momentum,
                                                     affine=layer_to_adjust.affine,
                                                     track_running_stats=layer_to_adjust.track_running_stats,
                                                     device=model_device)

                     # Copy running mean/var and affine parameters (if they exist)
                     if layer_to_adjust.track_running_stats:
                         new_layer.running_mean = layer_to_adjust.running_mean[keep_indices].clone()
                         new_layer.running_var = layer_to_adjust.running_var[keep_indices].clone()
                         new_layer.num_batches_tracked = layer_to_adjust.num_batches_tracked.clone() # Reset maybe? Copy for now.
                     if layer_to_adjust.affine:
                         new_layer.weight.data = layer_to_adjust.weight.data[keep_indices].clone()
                         new_layer.bias.data = layer_to_adjust.bias.data[keep_indices].clone()


            
            # If a new layer was created, replace the old one
            if new_layer is not None:
                _set_module_by_name(model, current_layer_name, new_layer)
                adjusted_layers.add(current_layer_name) # Mark as adjusted
                print(f"  Successfully adjusted and replaced '{current_layer_name}'.")
            else:
                 # This case happens if the layer type isn't handled or input dims already matched
                 if not isinstance(layer_to_adjust, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)):
                      print(f"Warning: Layer type {type(layer_to_adjust)} found after reduced layer '{previous_layer_name}'. Automatic input adjustment not implemented for this type. Manual adjustment might be needed.")


    print("--- Row Pruning Reduction Finished ---")
    return model




@torch.no_grad()
def column_based_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Column-based pruning using L1 norm, returning a mask for entire columns.
    Does NOT modify the input tensor. Designed primarily for 2D tensors (like Linear layers).

    For Conv layers ([out_ch, in_ch, kH, kW]), this prunes based on the L1 norm
    of slices along the input channel dimension (dim=1).

    :param tensor: torch.Tensor, weight matrix of a layer (e.g., shape [out_features, in_features])
    :param sparsity: float, pruning sparsity (fraction of columns/input channels to prune)
    :return: torch.Tensor, binary mask (1 = keep, 0 = prune) of the same shape as tensor
    """
    # Ensure tensor has at least 2 dimensions
    if tensor.dim() < 2:
        # Handle 1D or 0D tensors: cannot prune columns, return mask of ones
        print(f"Warning: Tensor dimension is {tensor.dim()}, cannot perform column pruning. Returning mask of ones.")
        return torch.ones_like(tensor)
        
    # Ensure sparsity is between 0 and 1
    sparsity = min(max(0.0, sparsity), 1.0)

    # Identify the column dimension (typically in_features or in_channels)
    col_dim = 1 # For Linear [out, in], Conv [out, in, H, W], etc.
    num_cols = tensor.shape[col_dim]
    num_prune = round(num_cols * sparsity)  # Number of columns to prune

    if num_prune == 0:
        return torch.ones_like(tensor, device=tensor.device)  # No pruning

    if num_prune >= num_cols:
        return torch.zeros_like(tensor, device=tensor.device) # Prune everything

    # --- Compute L1 norm for each column/input channel ---
    # Sum absolute values across all dimensions EXCEPT the column dimension
    dims_to_sum = list(range(tensor.dim()))
    dims_to_sum.remove(col_dim)
    col_l1_norms = torch.sum(torch.abs(tensor), dim=dims_to_sum) # Shape: [num_cols]

    # find indices to keep 
    num_keep = num_cols - num_prune
    
    # Find indices of top-k largest norms (columns to keep)
    _, keep_indices = torch.topk(col_l1_norms, k=num_keep, largest=True)

    # Create binary mask 
    # Start with a 1D mask for columns
    col_mask_1d = torch.zeros(num_cols, device=tensor.device, dtype=torch.float32)
    col_mask_1d[keep_indices] = 1.0 # Set kept columns to 1

    # Expand mask to the original tensor's shape 
    # Reshape the 1D mask to broadcast correctly along the column dimension
    # Shape needs to be [1, num_cols, 1, 1, ...] for broadcasting
    mask_shape = [1] * tensor.dim() # Create [1, 1, ...]
    mask_shape[col_dim] = num_cols # Set the column dimension size -> [1, num_cols, 1, ...]
    mask_to_expand = col_mask_1d.view(mask_shape) # Reshape the 1D mask

    # Expand the mask to the full shape of the original tensor
    mask = mask_to_expand.expand_as(tensor)

    # this function does NOT modify the input tensor in place.
    return mask.clone() # Return a clone of the mask





@torch.no_grad()
def col_based_prune_reduction(model: nn.Module, layers_to_reduce: list[str]) -> nn.Module:
    """
    Reduces the dimensions of specified layers after column-based pruning
    by removing zeroed columns (input features/channels) and adjusting the
    output dimensions of *preceding* layers accordingly.

    Assumes the model has already been pruned column-wise (e.g., using masks
    from `column_based_prune` applied via `Pruner.apply`).
    Modifies the model in place.

    Args:
        model (nn.Module): The PyTorch model (already column-pruned).
        layers_to_reduce (list[str]): List of layer names (weights) whose columns
                                      (input dim) should be physically removed

    Returns:
        nn.Module: The modified model with reduced layers.

    Raises:
        ValueError: If a layer type is unsupported or dimensions mismatch.
        AttributeError: If a layer name is not found.
        NotImplementedError: If automatic preceding layer adjustment fails.
    """
    
    model_device = next(model.parameters()).device
    module_list = list(model.named_modules()) # Get ordered list
    module_dict = OrderedDict(module_list) # For easier name lookup if needed

    # --- Pass 1: Identify columns to keep for each target layer ---
    # Store info about the reduction needed for each target layer
    col_reduction_info = {} # target_layer_name -> info
    print("--- Starting Column Pruning Reduction ---")
    print("Pass 1: Identifying columns to keep...")

    for layer_weight_name in layers_to_reduce:
        if not layer_weight_name.endswith('.weight'):
            print(f"Warning: Skipping '{layer_weight_name}'. Expecting layer weight name.")
            continue

        layer_name = layer_weight_name.replace('.weight', '')

        try:
            layer_module = _get_module_by_name(model, layer_name)
            layer_weight = layer_module.weight.data
        except AttributeError:
            print(f"Error: Layer '{layer_name}' not found in the model.")
            continue

        if not isinstance(layer_module, (nn.Linear, nn.Conv2d)):
             print(f"Warning: Layer type {type(layer_module)} for '{layer_name}' not currently supported for column reduction. Skipping.")
             continue

        # Identify non-zero columns (columns to keep)
        col_dim = 1 # Dimension corresponding to input features/channels
        if layer_weight.dim() < 2:
            print(f"Warning: Layer '{layer_name}' weight dimension is < 2. Skipping column reduction.")
            continue

        # Sum absolute values across all dimensions EXCEPT the column dimension
        dims_to_sum = list(range(layer_weight.dim()))
        dims_to_sum.remove(col_dim)
        col_sums = torch.sum(torch.abs(layer_weight), dim=dims_to_sum)

        keep_indices = torch.where(col_sums > 1e-8)[0] # Indices of columns/input channels to keep

        original_cols = layer_weight.shape[col_dim]
        new_cols = len(keep_indices) # This will be the new input dimension

        if new_cols == original_cols:
            print(f"Layer '{layer_name}' has no fully zeroed columns to remove. No reduction needed for this layer.")
            # Still might need adjustment if a *previous* layer's columns were pruned
            continue

        if new_cols == 0:
            print(f"Warning: All columns in layer '{layer_name}' are zero. Input dimension will be 0.")
            # This is problematic, model might become unusable.

        print(f"  Layer '{layer_name}': Found {new_cols} columns to keep out of {original_cols}.")

        # Store the necessary information
        col_reduction_info[layer_name] = {
            'keep_indices': keep_indices.to('cpu'), # Store on CPU to avoid device issues later if model moves
            'new_input_dim': new_cols,
            'original_input_dim': original_cols
        }

    # --- Pass 2: Modify layers (Target Layers and Preceding Layers) ---
    print("\nPass 2: Modifying layers...")
    adjusted_layers = set() # Keep track of layers already modified to prevent double adjustments

    # Iterate through modules using index for predecessor lookup
    for i, (current_layer_name, current_module) in enumerate(module_list):
        if current_layer_name in adjusted_layers or not current_layer_name:
            continue

        # --- Check 1: Is this layer a TARGET for column reduction? ---
        is_target = current_layer_name in col_reduction_info
        target_info = col_reduction_info.get(current_layer_name)

        # --- Check 2: Does this layer PRECEDE a target layer? ---
        precedes_target = None
        preceding_info = None # Info for the target layer it precedes
        # Look ahead in the sequence (simple assumption)
        if i + 1 < len(module_list):
             potential_next_layer_name = module_list[i+1][0]
             # More robust: search forward skipping non-parametric layers
             next_param_idx = -1
             for j in range(i + 1, len(module_list)):
                 nln, nlm = module_list[j]
                 if hasattr(nlm, 'weight') and isinstance(nlm, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)):
                     potential_next_layer_name = nln
                     next_param_idx = j
                     break

             if potential_next_layer_name in col_reduction_info:
                 # Check if current layer's output dimension matches the *original* input dim of the next target
                 next_target_info = col_reduction_info[potential_next_layer_name]
                 current_output_dim = -1
                 if isinstance(current_module, (nn.Linear)): current_output_dim = current_module.out_features
                 elif isinstance(current_module, (nn.Conv2d)): current_output_dim = current_module.out_channels
                 elif isinstance(current_module, (nn.BatchNorm1d, nn.BatchNorm2d)): current_output_dim = current_module.num_features

                 if current_output_dim == next_target_info['original_input_dim']:
                     precedes_target = potential_next_layer_name
                     preceding_info = next_target_info # Contains keep_indices for *next* layer's input

        # --- Apply Modifications ---
        # Priority: Modify preceding layers first, then target layers if not already handled implicitly

        # Modify CURRENT layer if it PRECEDES a target
        if precedes_target and preceding_info:
            print(f"Adjusting layer '{current_layer_name}' because it precedes target '{precedes_target}'...")
            keep_indices_for_output = preceding_info['keep_indices'].to(model_device)
            new_output_dim = preceding_info['new_input_dim'] # The new output dim for current = new input dim for next
            
            original_weight = current_module.weight.data
            has_bias = hasattr(current_module, 'bias') and current_module.bias is not None
            original_bias = current_module.bias.data if has_bias else None

            new_preceding_layer = None
            if isinstance(current_module, nn.Linear):
                original_output_dim = current_module.out_features
                if original_output_dim != new_output_dim: # Check if change needed
                    print(f"  Adjusting Linear layer '{current_layer_name}' (preceding): out_features {original_output_dim} -> {new_output_dim}")
                    # Slice ROWS (output features) using the keep_indices from the *next* layer's columns
                    new_weight = original_weight[keep_indices_for_output, :]
                    new_bias = original_bias[keep_indices_for_output] if has_bias else None

                    new_preceding_layer = nn.Linear(in_features=current_module.in_features,
                                                    out_features=new_output_dim,
                                                    bias=has_bias, device=model_device)
                else: print(f"  Skipping adjustment for '{current_layer_name}', output dimension already matches.")

            elif isinstance(current_module, nn.Conv2d):
                original_output_dim = current_module.out_channels
                if original_output_dim != new_output_dim: # Check if change needed
                     # Check groups
                    if current_module.groups != 1 and current_module.groups != original_output_dim:
                        print(f"Warning: Grouped convolution found in preceding layer '{current_layer_name}'. Automatic output channel reduction might be incorrect. Skipping adjustment.")
                    else:
                        print(f"  Adjusting Conv2d layer '{current_layer_name}' (preceding): out_channels {original_output_dim} -> {new_output_dim}")
                        # Slice OUTPUT CHANNELS (dim 0) using the keep_indices from the *next* layer's input channels
                        new_weight = original_weight[keep_indices_for_output, :, :, :]
                        new_bias = original_bias[keep_indices_for_output] if has_bias else None
                        
                        # If it was depthwise, groups must match new output dim
                        new_groups = current_module.groups if current_module.groups==1 else new_output_dim

                        new_preceding_layer = nn.Conv2d(in_channels=current_module.in_channels,
                                                        out_channels=new_output_dim,
                                                        kernel_size=current_module.kernel_size,
                                                        stride=current_module.stride,
                                                        padding=current_module.padding,
                                                        dilation=current_module.dilation,
                                                        groups=new_groups,
                                                        bias=has_bias,
                                                        padding_mode=current_module.padding_mode,
                                                        device=model_device)
                else: print(f"  Skipping adjustment for '{current_layer_name}', output dimension already matches.")
            
            # If the layer *after* this preceding layer is BatchNorm, it also needs adjustment
            potential_bn_idx = i + 1
            if potential_bn_idx < len(module_list):
                 bn_layer_name, bn_module = module_list[potential_bn_idx]
                 if isinstance(bn_module, (nn.BatchNorm1d, nn.BatchNorm2d)) and bn_layer_name not in adjusted_layers:
                     original_bn_features = bn_module.num_features
                     if original_bn_features != new_output_dim:
                          print(f"  Adjusting BatchNorm layer '{bn_layer_name}' following preceding layer: num_features {original_bn_features} -> {new_output_dim}")
                          if isinstance(bn_module, nn.BatchNorm1d):
                              new_bn_layer = nn.BatchNorm1d(new_output_dim, eps=bn_module.eps, momentum=bn_module.momentum, affine=bn_module.affine, track_running_stats=bn_module.track_running_stats, device=model_device)
                          else:
                              new_bn_layer = nn.BatchNorm2d(new_output_dim, eps=bn_module.eps, momentum=bn_module.momentum, affine=bn_module.affine, track_running_stats=bn_module.track_running_stats, device=model_device)
                              
                          # Copy relevant stats/params
                          if bn_module.track_running_stats:
                              new_bn_layer.running_mean = bn_module.running_mean[keep_indices_for_output].clone()
                              new_bn_layer.running_var = bn_module.running_var[keep_indices_for_output].clone()
                              new_bn_layer.num_batches_tracked = bn_module.num_batches_tracked.clone()
                          if bn_module.affine:
                              new_bn_layer.weight.data = bn_module.weight.data[keep_indices_for_output].clone()
                              new_bn_layer.bias.data = bn_module.bias.data[keep_indices_for_output].clone()
                              
                          _set_module_by_name(model, bn_layer_name, new_bn_layer)
                          adjusted_layers.add(bn_layer_name)
                     else: print(f"  Skipping adjustment for '{bn_layer_name}', features already match.")

            # Replace the preceding layer if needed
            if new_preceding_layer is not None:
                new_preceding_layer.weight.data = new_weight
                if has_bias and new_bias is not None:
                    new_preceding_layer.bias.data = new_bias
                _set_module_by_name(model, current_layer_name, new_preceding_layer)
                adjusted_layers.add(current_layer_name)

        # 2. Modify CURRENT layer if it is a TARGET layer
        #    (Only if it wasn't already handled as a preceding layer, which is unlikely for Conv/Linear)
        if is_target and current_layer_name not in adjusted_layers:
            print(f"Adjusting layer '{current_layer_name}' as a target for column reduction...")
            keep_indices_for_input = target_info['keep_indices'].to(model_device)
            new_input_dim = target_info['new_input_dim']
            original_input_dim = target_info['original_input_dim']

            if original_input_dim != new_input_dim: # Check if change needed
                original_weight = current_module.weight.data
                has_bias = hasattr(current_module, 'bias') and current_module.bias is not None
                original_bias = current_module.bias.data if has_bias else None # Bias is UNCHANGED by input dim change

                new_target_layer = None
                if isinstance(current_module, nn.Linear):
                    print(f"  Adjusting Linear layer '{current_layer_name}' (target): in_features {original_input_dim} -> {new_input_dim}")
                    # Slice COLUMNS (input features)
                    new_weight = original_weight[:, keep_indices_for_input]

                    new_target_layer = nn.Linear(in_features=new_input_dim,
                                                 out_features=current_module.out_features, # Output dim is unchanged
                                                 bias=has_bias, device=model_device)

                elif isinstance(current_module, nn.Conv2d):
                    # Check groups relative to *input* channels
                    if current_module.groups != 1 and current_module.groups != original_input_dim:
                         print(f"Warning: Grouped convolution found in target layer '{current_layer_name}'. Automatic input channel reduction might be incorrect. Skipping adjustment.")
                    else:
                        print(f"  Adjusting Conv2d layer '{current_layer_name}' (target): in_channels {original_input_dim} -> {new_input_dim}")
                        # Slice INPUT CHANNELS (dim 1)
                        new_weight = original_weight[:, keep_indices_for_input, :, :]
                        
                        # If it was depthwise (groups == in_channels), new groups must match new input dim
                        new_groups = current_module.groups if current_module.groups==1 else new_input_dim
                        
                        new_target_layer = nn.Conv2d(in_channels=new_input_dim,
                                                     out_channels=current_module.out_channels, # Output dim is unchanged
                                                     kernel_size=current_module.kernel_size,
                                                     stride=current_module.stride,
                                                     padding=current_module.padding,
                                                     dilation=current_module.dilation,
                                                     groups=new_groups,
                                                     bias=has_bias,
                                                     padding_mode=current_module.padding_mode,
                                                     device=model_device)

                # Replace the target layer if needed
                if new_target_layer is not None:
                    new_target_layer.weight.data = new_weight
                    if has_bias and original_bias is not None:
                        new_target_layer.bias.data = original_bias # Use original bias
                    _set_module_by_name(model, current_layer_name, new_target_layer)
                    adjusted_layers.add(current_layer_name)
            else:
                 print(f"  Skipping adjustment for '{current_layer_name}', input dimension already matches expected.")


    print("--- Column Pruning Reduction Finished ---")
    return model



# ------ END OF FC PRUNING FUNCTIONS ------ 




# ------ START OF CHANNEL PRUNING HELPER FUNCTIONS ------ 

def get_input_channel_importance(weight, device, is_depthwise=False):
    """
    Compute the importance of input channels in a convolutional layer.
    For standard convolutions, importance is computed based on L2 norm of input channels.
    For depthwise convolutions, importance is computed per output channel (since in_channels == out_channels).
    """
    in_channels = weight.shape[1]  # Number of input channels
    
    if is_depthwise:
        # For depthwise convs, importance is per filter (out_channels == in_channels)
        importances = torch.norm(weight.detach(), p=2, dim=(1, 2, 3))  # Norm across (kernel_h, kernel_w)
    else:
        # For normal convs, importance is per input channel
        importances = []
        for i_c in range(in_channels):
            channel_weight = weight.detach()[:, i_c]  # Extract all output filters for this input channel
            importance = torch.norm(channel_weight, p=2)  # L2 norm
            importances.append(importance.view(1))

        importances = torch.cat(importances)

    return importances.to(weight.device)




def extract_non_depthwise_conv_bn_pairs(module):
    """
    Extracts all (Conv2d, BatchNorm2d) pairs where:
    - Conv2d is NOT a depthwise convolution (groups=1)
    - BatchNorm2d immediately follows the Conv2d layer
    Returns two lists: one for Conv2d layers and one for BatchNorm2d layers.
    """
    conv_layers = []
    bn_layers = []
    
    previous_layer = None  # Track the last Conv2d layer

    for child in module.children():
        if isinstance(child, nn.Conv2d) and child.groups == 1:  # Ignore depthwise convolutions
            previous_layer = child  # Store non-depthwise Conv2d layer
        elif isinstance(child, nn.BatchNorm2d) and previous_layer is not None:
            # Pair Conv2d with BatchNorm2d
            conv_layers.append(previous_layer)
            bn_layers.append(child)
            previous_layer = None  # Reset after pairing
        else:
            # Recursively search submodules
            sub_conv, sub_bn = extract_non_depthwise_conv_bn_pairs(child)
            conv_layers.extend(sub_conv)
            bn_layers.extend(sub_bn)

    return conv_layers, bn_layers



def extract_all_conv_bn_pairs(module):
    """
    Extracts all (Conv2d, BatchNorm2d) pairs, including depthwise convolutions.
    - Conv2d layers (both standard and depthwise) are included.
    - BatchNorm2d immediately following a Conv2d layer is considered its pair.

    Returns two lists: one for Conv2d layers and one for BatchNorm2d layers.
    """
    conv_layers = []
    bn_layers = []
    
    previous_layer = None  # Track the last Conv2d layer

    for child in module.children():
        if isinstance(child, nn.Conv2d):  # Include both depthwise and standard convolutions
            previous_layer = child  # Store Conv2d layer
        elif isinstance(child, nn.BatchNorm2d) and previous_layer is not None:
            # Pair Conv2d with BatchNorm2d
            conv_layers.append(previous_layer)
            bn_layers.append(child)
            previous_layer = None  # Reset after pairing
        else:
            # Recursively search submodules
            sub_conv, sub_bn = extract_all_conv_bn_pairs(child)
            conv_layers.extend(sub_conv)
            bn_layers.extend(sub_bn)

    return conv_layers, bn_layers


def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    """A function to calculate the number of layers to PRESERVE after pruning"""
    if channels == 0: return 0
    keep = int(round((1.0 - prune_ratio) * channels))
    if channels > 0 and keep == 0 and prune_ratio < 1.0:
       keep = 1
    return keep




def replace_layer(model, old_layer, new_layer):
    """
    Recursively replace a layer in the model (using object identity).
    Might be unreliable for complex models.
    """
    for name, module in model.named_children():
        if module is old_layer:
            # Find the full path for better logging if possible (difficult with only object)
            # Simple version:
            print(f"Set layer '{name}' in {type(model).__name__} to {new_layer} (using replace_layer)")
            setattr(model, name, new_layer)
            return True # Stop search once replaced
        # Recursively search in children
        elif len(list(module.children())) > 0:
             if replace_layer(module, old_layer, new_layer):
                 return True # Stop searching if replaced in grandchild
    return False


def find_module_parent_and_name(root_module, target_module):
    """Finds the parent and attribute name of a target module instance."""
    # Check root module's direct children first
    for child_name, child_module in root_module.named_children():
        if child_module is target_module:
             return root_module, child_name

    # Then check descendants
    for name, module in root_module.named_modules():
        # Skip the root module itself which was checked above
        if module is root_module:
            continue
        for child_name, child_module in module.named_children():
            if child_module is target_module:
                return module, child_name # Found parent and attribute name
    return None, None # Not found



# ------ END OF CHANNEL PRUNING HELPER FUNCTIONS ------ 



# ------ START OF CHANNEL PRUNING FUNCTIONS ------ 


@torch.no_grad()
def apply_conv2d_sorting(model, device, verbose=False):
    model = copy.deepcopy(model)  # Avoid modifying the original model
    all_convs, all_bns = [], []

    all_convs, all_bns = extract_non_depthwise_conv_bn_pairs(model)
    
    for i_conv in range(len(all_convs) - 1):
        prev_conv = all_convs[i_conv]
        prev_bn = all_bns[i_conv] if i_conv < len(all_bns) else None
        next_conv = all_convs[i_conv + 1]

        #handle logic for depthwise layers seperately 
        is_depthwise = prev_conv.groups == prev_conv.in_channels
        next_is_depthwise = next_conv.groups == next_conv.in_channels
        
        if is_depthwise or next_is_depthwise:
            print(f"Warning: Conv {i_conv} is depthwise. Skipping.")
            continue 

        if verbose:
            print(f"\nProcessing Conv Layer {i_conv}: {prev_conv}")
            print(f"Next Conv: {next_conv}")

        # Compute importance correctly based on type
        importance = get_input_channel_importance(
            next_conv.weight, device, is_depthwise=next_is_depthwise
        )

        # Sort importance
        sort_idx = torch.argsort(importance, descending=True)

        if verbose:
            print(f"\nImportance: {importance}")
            print(f"Sort Index: {sort_idx}")

        # Ensure valid sorting indices
        if sort_idx.shape[0] != prev_conv.out_channels:
            print(f"Warning: Mismatch in sorting indices for Conv {i_conv}. Skipping.")
            print("Sort idx shape", sort_idx.shape)
            print("prev_conv.out_channels", prev_conv.out_channels)
            continue

        # Sort output channels of prev_conv
        prev_conv.weight.copy_(torch.index_select(prev_conv.weight.detach().clone(), 0, sort_idx))

        # Sort BN layer if it exists
        if prev_bn is not None:
            for tensor_name in ['weight', 'bias', 'running_mean', 'running_var']:
                tensor_to_apply = getattr(prev_bn, tensor_name)
                if tensor_to_apply.shape[0] == sort_idx.shape[0]:
                    tensor_to_apply.copy_(torch.index_select(tensor_to_apply.detach().clone(), 0, sort_idx))

        # Sorting next conv’s input channels (idx 1)
        if not next_is_depthwise and next_conv.weight.shape[1] == sort_idx.shape[0]:
            next_conv.weight.copy_(torch.index_select(next_conv.weight.detach().clone(), 1, sort_idx))
        else:
            print(f"Warning: Mismatch in weight configurations for Conv {i_conv}. Skipping.")
            
    return model



@torch.no_grad()
def uniform_channel_prune(model: nn.Module, prune_ratio, SqueezeExcitationModuleType, verbose=False) -> nn.Module:
    """
    Apply channel pruning using the original loop structure.
    Adds handling for SE blocks AND the subsequent point_linear conv layer (FIXED LOCATION).
    Includes fix for depthwise weight copy.
    Uses object-based replace_layer.
    """

    assert isinstance(prune_ratio, (float, list))
    assert issubclass(SqueezeExcitationModuleType, nn.Module), "SqueezeExcitationModuleType must be provided and be a nn.Module subclass."

    model = copy.deepcopy(model)

    all_convs_initial, all_bns_initial = extract_all_conv_bn_pairs(model) 
    if not all_convs_initial:
        print("Warning: Initial extraction found no Conv-BN pairs.")
        return model

    if isinstance(prune_ratio, list):
        if len(prune_ratio) != len(all_convs_initial):
             print(f"Warning: Length of prune_ratio list ({len(prune_ratio)}) != number of Conv-BN pairs found ({len(all_convs_initial)}). ")
             return
        prune_ratios = prune_ratio
    else:
        prune_ratios = [prune_ratio] * len(all_convs_initial)

    num_iterations = min(len(all_convs_initial), len(all_bns_initial))
    if len(all_convs_initial) != len(all_bns_initial):
         print(f"Warning: Mismatch conv/bn counts ({len(all_convs_initial)}/{len(all_bns_initial)}). Iterating {num_iterations} times.")

    # --- Main Pruning Loop ---
    for i_pair in range(num_iterations):
        all_convs, all_bns = extract_all_conv_bn_pairs(model)
        if i_pair >= len(all_convs) or i_pair >= len(all_bns):
             print(f"Warning: Index {i_pair} out of bounds after re-extracting. Stopping early.")
             break

        prev_conv = all_convs[i_pair]
        prev_bn = all_bns[i_pair]
        next_conv_from_list = all_convs[i_pair + 1] if i_pair + 1 < len(all_convs) else None
        p_ratio = prune_ratios[i_pair]

        if verbose:
            print(f"\n--- Processing Pair {i_pair} ---")
            _, prev_conv_name = find_module_parent_and_name(model, prev_conv)
            _, prev_bn_name = find_module_parent_and_name(model, prev_bn)
            print(f"prev_conv ({prev_conv_name or '?'}): {prev_conv}")
            print(f"prev_bn ({prev_bn_name or '?'}): {prev_bn}")
            print(f"next_conv (from list): {next_conv_from_list}")

        original_channels = prev_conv.out_channels
        original_in_channels = prev_conv.in_channels
        n_keep = get_num_channels_to_keep(original_channels, p_ratio)

        if n_keep == original_channels or n_keep <= 0:
            if verbose: print(f"Skipping pruning for pair {i_pair}: original={original_channels}, n_keep={n_keep}.")
            continue

        if verbose: print(f"Pruning {original_channels} -> {n_keep} channels.")

        # Create new prev_conv
        is_depthwise = prev_conv.groups == original_in_channels and original_in_channels > 1
        if verbose:
            print("Is depthwise: ", is_depthwise)
    

        #----  Adjust n_keep for depthwise by finding the CLOSEST divisor ----
        # This is needed cause in the i-1 iteration, next_conv (now this prev_conv) input channels was pruned to be n_keep_i-1 to match 
        #Now that prev_conv is a depthwise layer, we can no longer change its number of input channels since it would break the last later's output channels
        if is_depthwise and original_in_channels > 1: # Avoid division by zero/mod zero
            initial_n_keep = n_keep # Store the target
            if original_in_channels % n_keep != 0:
                if verbose:
                    print(f"    Adjusting n_keep ({n_keep}) for DW divisibility (in={original_in_channels}). Finding closest divisor.")
                # Find divisors
                divisors = []
                for i in range(1, int(original_in_channels**0.5) + 1):
                    if original_in_channels % i == 0:
                        divisors.append(i)
                        if i*i != original_in_channels:
                            divisors.append(original_in_channels // i)
                divisors.sort()

                if not divisors: # Should not happen if original_in_channels > 1
                    print(f"ERROR: No divisors found for {original_in_channels}. Setting n_keep=1.")
                    n_keep = 1
                else:
                    # Find the divisor closest to the initial_n_keep
                    closest_divisor = divisors[0]
                    min_diff = abs(initial_n_keep - closest_divisor)
                    for d in divisors[1:]:
                        diff = abs(initial_n_keep - d)
                        # Let's prioritize the one closer, break ties favoring larger divisor maybe
                        if diff < min_diff:
                            min_diff = diff
                            closest_divisor = d
                        elif diff == min_diff and d > closest_divisor: # Tie-break towards larger divisor
                            closest_divisor = d

                    n_keep = closest_divisor
                    # Ensure n_keep is at least 1
                    n_keep = max(1, n_keep)

                    if verbose:
                        print(f"    Target n_keep was {initial_n_keep}. Closest divisor of {original_in_channels} is {n_keep}.")
            # else: n_keep already divides or not depthwise

        # Calculate new_groups based on the potentially adjusted n_keep
        new_groups = n_keep if is_depthwise else prev_conv.groups
        
        # --- END of Adjustment---


        print("Creating new prev_conv layer...")
        new_prev_conv = nn.Conv2d(
            in_channels=original_in_channels, out_channels=n_keep,
            kernel_size=prev_conv.kernel_size, stride=prev_conv.stride,
            padding=prev_conv.padding, dilation=prev_conv.dilation,
            groups=new_groups, bias=(prev_conv.bias is not None)
        )

        # Copy Weights (with depthwise fix)
        if prev_conv.weight is not None:
            if is_depthwise:
                source_weight_slice = prev_conv.weight.data[:n_keep, :, :, :]
                target_channels_per_group = new_prev_conv.weight.shape[1]
                if target_channels_per_group > 0:
                     target_weight = source_weight_slice.repeat(1, target_channels_per_group, 1, 1)
                     if target_weight.shape == new_prev_conv.weight.shape:
                          new_prev_conv.weight.data.copy_(target_weight)
                     else:
                          print(f"ERROR: Shape mismatch DW weight copy {i_pair}. Expected {new_prev_conv.weight.shape}, got {target_weight.shape}.")
                          new_prev_conv.weight.data.copy_(prev_conv.weight.data[:n_keep, :new_prev_conv.weight.shape[1], ...])
                else:
                     print(f"ERROR: Zero target channels/group DW copy {i_pair}.")
            else: # Standard/Grouped Conv
                new_prev_conv.weight.data.copy_(prev_conv.weight.data[:n_keep, ...])
        if prev_conv.bias is not None and new_prev_conv.bias is not None:
             new_prev_conv.bias.data.copy_(prev_conv.bias.data[:n_keep])

        # Create new BatchNorm2d layer
        print("Creating new prev_bn layer...")
        new_prev_bn = nn.BatchNorm2d(
            n_keep, eps=prev_bn.eps, momentum=prev_bn.momentum,
            affine=prev_bn.affine, track_running_stats=prev_bn.track_running_stats)
        # Copy BN parameters
        if prev_bn.affine:
            if prev_bn.weight is not None: new_prev_bn.weight.data.copy_(prev_bn.weight.data[:n_keep])
            if prev_bn.bias is not None: new_prev_bn.bias.data.copy_(prev_bn.bias.data[:n_keep])
        if prev_bn.track_running_stats:
            if prev_bn.running_mean is not None: new_prev_bn.running_mean.data.copy_(prev_bn.running_mean.data[:n_keep])
            if prev_bn.running_var is not None: new_prev_bn.running_var.data.copy_(prev_bn.running_var.data[:n_keep])
            if hasattr(prev_bn, 'num_batches_tracked') and prev_bn.num_batches_tracked is not None:
                 new_prev_bn.num_batches_tracked.data.copy_(prev_bn.num_batches_tracked.data)

        # --- Attempt to Find and Handle SE Block AND subsequent point_linear ---
        se_handled = False
        point_linear_after_se_handled = False
        original_point_linear_conv_after_se = None
        new_point_linear_conv_after_se = None

        # Find parent of prev_bn (e.g., the depth_conv Sequential)
        parent_module, _ = find_module_parent_and_name(model, prev_bn)

        if parent_module is not None:
            # Find the parent of the parent (the grandparent, e.g., mobile_inverted_conv block)
            # SE modules are often a sibling inside the parent, and point_linear layers are a sibling inside the grandparent.

            grandparent_module, parent_attr_name = find_module_parent_and_name(model, parent_module)

            # Check for sibling 'se' block *within the immediate parent*
            if hasattr(parent_module, 'se'):
                se_module_candidate = parent_module.se
                if isinstance(se_module_candidate, SqueezeExcitationModuleType):
                    se_module = se_module_candidate
                    reduce_attr, expand_attr = 'fc.reduce', 'fc.expand'
                    try:
                        reduce_layer = getattr(getattr(se_module, 'fc', None), 'reduce', None)
                        expand_layer = getattr(getattr(se_module, 'fc', None), 'expand', None)

                        if isinstance(reduce_layer, nn.Conv2d) and isinstance(expand_layer, nn.Conv2d):
                            original_se_input_channels = reduce_layer.in_channels # Store original expected input

                            # Check if SE block's input matches the layer being pruned's ORIGINAL output
                            if True:
                                print(f"  Handling SE Block (within {type(parent_module).__name__} named '{parent_attr_name or '?'}').")
                                se_handled = True

                                # Prune SE Reduce Layer Input
                                bottleneck_ch = reduce_layer.out_channels
                                print(f"    Modifying SE reduce input: {original_se_input_channels} -> {n_keep}")

                                new_reduce_layer = nn.Conv2d(n_keep, bottleneck_ch, 1, 1, 0, bias=(reduce_layer.bias is not None))

                                new_reduce_layer.weight.data.copy_(reduce_layer.weight.data[:, :n_keep, ...])
                                new_reduce_layer.bias.data.copy_(reduce_layer.bias.data)
                                se_module.fc.reduce = new_reduce_layer # Direct replacement

                                # Prune SE Expand Layer Output
                                print(f"    Modifying SE expand output: {expand_layer.out_channels} -> {n_keep}")
                                new_expand_layer = nn.Conv2d(bottleneck_ch, n_keep, 1, 1, 0, bias=(expand_layer.bias is not None))
                                new_expand_layer.weight.data.copy_(expand_layer.weight.data[:n_keep, :, ...])
                                new_expand_layer.bias.data.copy_(expand_layer.bias.data[:n_keep])

                                se_module.fc.expand = new_expand_layer # Direct replacement

                                # Look for point_linear in the GRANDPARENT module ---
                                if grandparent_module is not None:
                                    point_linear_attr = 'point_linear'
                                    if hasattr(grandparent_module, point_linear_attr):
                                        point_linear_module = getattr(grandparent_module, point_linear_attr)
                                        consumer_conv = None
                                        # Try finding the Conv2d layer inside point_linear module
                                        if isinstance(point_linear_module, nn.Sequential) and len(point_linear_module) > 0 and isinstance(point_linear_module[0], nn.Conv2d):
                                             consumer_conv = point_linear_module[0]
                                        elif hasattr(point_linear_module, 'conv') and isinstance(point_linear_module.conv, nn.Conv2d):
                                             consumer_conv = point_linear_module.conv
                                        else:
                                             if verbose: print(f"    Found '{point_linear_attr}' in grandparent, but cannot find expected Conv2d inside.")

                                        if consumer_conv is not None:
                                            # Check if its input matches the ORIGINAL input channels of the SE block
                                            if consumer_conv.in_channels == original_se_input_channels:
                                                print(f"    Modifying '{point_linear_attr}' conv input: {original_se_input_channels} -> {n_keep}")
                                                new_point_linear_conv_after_se = nn.Conv2d(
                                                    in_channels=n_keep, out_channels=consumer_conv.out_channels,
                                                    kernel_size=consumer_conv.kernel_size, stride=consumer_conv.stride,
                                                    padding=consumer_conv.padding, dilation=consumer_conv.dilation,
                                                    groups=consumer_conv.groups, bias=(consumer_conv.bias is not None) )
                                                if consumer_conv.weight is not None: new_point_linear_conv_after_se.weight.data.copy_(consumer_conv.weight.data[:, :n_keep, ...])
                                                if consumer_conv.bias is not None and new_point_linear_conv_after_se.bias is not None: new_point_linear_conv_after_se.bias.data.copy_(consumer_conv.bias.data)
                                                original_point_linear_conv_after_se = consumer_conv
                                                point_linear_after_se_handled = True
                                            else:
                                                 if verbose: print(f"    Input of '{point_linear_attr}' conv ({consumer_conv.in_channels}) != original SE input ({original_se_input_channels}). Skipping.")
                                    else:
                                         if verbose: print(f"    SE handled, but no sibling '{point_linear_attr}' found in grandparent module '{type(grandparent_module).__name__}'.")
                                else: # Grandparent not found
                                    if verbose: print(f"    SE handled, but could not find grandparent module to look for '{point_linear_attr}'.")
                            else: # SE input check failed
                                 if verbose: print(f"  SE block found, but reduce layer in_channels ({original_se_input_channels}) != prev_conv output channels ({original_channels}). Skipping.")
                        else: # Not Conv2d inside SE fc
                             if verbose: print(f"  SE block found, but layers '{reduce_attr}'/'{expand_attr}' not found or not Conv2d.")
                    except AttributeError: # Failed fc.reduce/expand access
                         if verbose: print(f"  SE block found, but failed to access expected structure '{reduce_attr}'/'{expand_attr}'.")
                # else: sibling 'se' exists but is not the correct type

        # --- Handle modification of the *next* conv layer's INPUT channels (Original Logic - ONLY IF SE WASN'T HANDLED) ---
        new_next_conv_from_list = None
        modified_next_conv_list = False

        if not se_handled and next_conv_from_list is not None:
            modified_next_conv_list = True
            print("SE block not detected/handled. Applying original logic to next_conv from list.")
            # NOTE: Check `if next_conv_from_list.in_channels != original_channels:` was removed previously
            # Create new_next_conv_from_list based on original logic

            is_next_depthwise = next_conv_from_list.groups == next_conv_from_list.in_channels and next_conv_from_list.in_channels > 1
            if is_next_depthwise:
                print("Handling next depthwise conv (from list):")
                
                new_next_conv_from_list = nn.Conv2d(
                    in_channels=n_keep, out_channels=n_keep, kernel_size=next_conv_from_list.kernel_size, stride=next_conv_from_list.stride,
                    padding=next_conv_from_list.padding, dilation=next_conv_from_list.dilation, groups=n_keep, bias=(next_conv_from_list.bias is not None))
                
                if next_conv_from_list.weight is not None: new_next_conv_from_list.weight.data.copy_(next_conv_from_list.weight.data[:n_keep, ...])
                if next_conv_from_list.bias is not None and new_next_conv_from_list.bias is not None: new_next_conv_from_list.bias.data.copy_(next_conv_from_list.bias.data[:n_keep])


            elif next_conv_from_list.groups == 1 or next_conv_from_list.in_channels % next_conv_from_list.groups == 0:
                 print("Handling next standard/grouped conv input (from list):")

                 if next_conv_from_list.groups > 1 and n_keep % next_conv_from_list.groups != 0:
                     print(f"ERROR: Cannot prune input of next grouped conv (from list) to {n_keep} (groups={next_conv_from_list.groups}). Skipping.")
                     modified_next_conv_list = False
                 else:

                    new_next_conv_from_list = nn.Conv2d(
                        in_channels=n_keep, out_channels=next_conv_from_list.out_channels, kernel_size=next_conv_from_list.kernel_size, stride=next_conv_from_list.stride,
                        padding=next_conv_from_list.padding, dilation=next_conv_from_list.dilation, groups=next_conv_from_list.groups, bias=(next_conv_from_list.bias is not None))
                    
                    if next_conv_from_list.weight is not None:
                         if next_conv_from_list.groups > 1: new_next_conv_from_list.weight.data.copy_(next_conv_from_list.weight.data[:, :(n_keep // next_conv_from_list.groups), ...])
                         else: new_next_conv_from_list.weight.data.copy_(next_conv_from_list.weight.data[:, :n_keep, ...])
                    
                    if next_conv_from_list.bias is not None and new_next_conv_from_list.bias is not None: new_next_conv_from_list.bias.data.copy_(next_conv_from_list.bias.data)
            else:
                 print(f"Warning: Cannot handle next_conv_from_list groups={next_conv_from_list.groups}/channels={next_conv_from_list.in_channels}. Skipping modification.")
                 modified_next_conv_list = False
        elif se_handled:
             if verbose: print("SE block handled, skipping modification of next_conv_from_list.")


        # --- Replace layers in the model ---
        print("Replacing layers...")
        replace_success_prev_conv = replace_layer(model, prev_conv, new_prev_conv)
        replace_success_prev_bn = replace_layer(model, prev_bn, new_prev_bn)

        replace_success_next_conv_list = True
        if modified_next_conv_list and new_next_conv_from_list is not None and next_conv_from_list is not None:
            replace_success_next_conv_list = replace_layer(model, next_conv_from_list, new_next_conv_from_list)
            if not replace_success_next_conv_list: print(f"ERROR: Failed to replace next_conv_from_list at pair {i_pair}.")

        replace_success_point_linear = True
        if point_linear_after_se_handled and new_point_linear_conv_after_se is not None:
            if original_point_linear_conv_after_se is not None:
                 replace_success_point_linear = replace_layer(model, original_point_linear_conv_after_se, new_point_linear_conv_after_se)
                 if not replace_success_point_linear: print(f"ERROR: Failed to replace point_linear conv after SE at pair {i_pair}.")
            else:
                 print(f"ERROR: Cannot replace point_linear conv after SE - original object reference is None at pair {i_pair}.")
                 replace_success_point_linear = False

        if not replace_success_prev_conv or not replace_success_prev_bn:
             print(f"ERROR: Failed basic layer replacement at pair {i_pair}. Model state might be inconsistent.")
             # break # Optionally stop the loop on critical failure

    print("\nPruning loop finished.")
    return model

# ------ END OF CHANNEL PRUNING FUNCTIONS ------ 


# --- Function to Prune a Single MobileInvertedResidualBlock ---

@torch.no_grad()
def prune_mb_inverted_block(
    model: nn.Module,
    target_conv_path: str, # Path to a conv within  the block (e.g., 'blocks.3.mobile_inverted_conv.inverted_bottleneck.conv')
    prune_ratio: float,
    MobileInvertedBlockClass: type, # The class type of your block (e.g., MobileInvertedResidualBlock)
    SqueezeExcitationModuleType: type, # The class type of your SE block
    verbose: bool = False
) -> nn.Module:
    """
    Prunes the intermediate expanded channels within a specific MobileInvertedResidualBlock.

    Args:
        model: The model containing the block.
        target_conv_path: String path to any conv layer inside the target block.
                          Used to identify the block to prune.
        prune_ratio: The ratio of *expanded* channels to remove (0.0 to 1.0).
        MobileInvertedBlockClass: The Python class of the main block (e.g., MobileInvertedResidualBlock).
        SqueezeExcitationModuleType: The Python class of the SE block.
        verbose: Print detailed information.

    Returns:
        The modified model (or original if block/layers not found).
    """
    assert isinstance(target_conv_path, str), "target_conv_path must be a string."
    assert isinstance(prune_ratio, float) and 0.0 <= prune_ratio < 1.0, "prune_ratio must be 0.0 <= ratio < 1.0"
    assert issubclass(MobileInvertedBlockClass, nn.Module)
    assert issubclass(SqueezeExcitationModuleType, nn.Module)

    model = copy.deepcopy(model)

    # 1. Find the Target Block Object and its Path
    block_to_prune = None
    block_path = None
    temp_module = model
    path_parts = target_conv_path.split('.')
    current_path = []
    for i, part in enumerate(path_parts):
        current_path.append(part)
        potential_module = None
        try:
             # Handle list indexing
             if part.isdigit() and isinstance(temp_module, (nn.ModuleList, nn.Sequential)):
                 potential_module = temp_module[int(part)]
             elif hasattr(temp_module, part):
                 potential_module = getattr(temp_module, part)
        except (IndexError, AttributeError):
             break # Invalid path part

        if potential_module is None: break

        if isinstance(potential_module, MobileInvertedBlockClass):
            block_to_prune = potential_module
            block_path = ".".join(current_path)
            if verbose: print(f"Found target block '{block_path}' of type {type(block_to_prune).__name__}")
            break
        temp_module = potential_module

    if block_to_prune is None:
        print(f"ERROR: Could not find MobileInvertedBlockClass instance containing path '{target_conv_path}'.")
        return model

    # 2. Identify Key Layers within the Block 
    try:
        expand_conv = block_to_prune.mobile_inverted_conv.inverted_bottleneck.conv
        expand_bn   = block_to_prune.mobile_inverted_conv.inverted_bottleneck.bn
        dw_conv     = block_to_prune.mobile_inverted_conv.depth_conv.conv
        dw_bn       = block_to_prune.mobile_inverted_conv.depth_conv.bn
        # SE is optional
        se_module   = getattr(getattr(block_to_prune.mobile_inverted_conv, 'depth_conv', None), 'se', None)
        point_conv  = block_to_prune.mobile_inverted_conv.point_linear.conv
        # Pointwise BN is not modified by this prune step
    except AttributeError as e:
        print(f"ERROR: Block '{block_path}' missing expected sub-layer structure: {e}. Cannot prune.")
        return model

    # Check types
    if not isinstance(expand_conv, nn.Conv2d) or \
       not isinstance(expand_bn, nn.BatchNorm2d) or \
       not isinstance(dw_conv, nn.Conv2d) or \
       not isinstance(dw_bn, nn.BatchNorm2d) or \
       not isinstance(point_conv, nn.Conv2d):
        print(f"ERROR: Incorrect layer types found within block '{block_path}'. Cannot prune.")
        return model

    has_se = se_module is not None and isinstance(se_module, SqueezeExcitationModuleType)
    reduce_layer, expand_layer = None, None
    if has_se:
        try: # Check SE structure
            reduce_layer = getattr(getattr(se_module, 'fc', None), 'reduce', None)
            expand_layer = getattr(getattr(se_module, 'fc', None), 'expand', None)
            if not isinstance(reduce_layer, nn.Conv2d) or not isinstance(expand_layer, nn.Conv2d):
                 print(f"Warning: SE block in '{block_path}' found, but internal fc.reduce/expand are not Conv2d. SE block will not be pruned.")
                 has_se = False # Treat as if no SE for pruning purposes
        except AttributeError:
            print(f"Warning: SE block in '{block_path}' found, but lacks expected fc.reduce/expand structure. SE block will not be pruned.")
            has_se = False


    # 3. Determine Pruning Dimension
    original_expanded_channels = expand_conv.out_channels
    if verbose: print(f"Original expanded channels: {original_expanded_channels}")

    # Check if layers dependent on expanded channels match
    if dw_conv.in_channels != original_expanded_channels or \
       dw_conv.out_channels != original_expanded_channels or \
       dw_conv.groups != original_expanded_channels or \
       dw_bn.num_features != original_expanded_channels or \
       point_conv.in_channels != original_expanded_channels or \
       (has_se and reduce_layer.in_channels != original_expanded_channels) or \
       (has_se and expand_layer.out_channels != original_expanded_channels):
        print(f"ERROR: Inconsistent channel dimensions found within block '{block_path}' relative to expansion conv output ({original_expanded_channels}). Cannot prune reliably.")
        return model

    # 4. Calculate n_keep and Indices based on Expansion BN
    if expand_bn.weight is None:
         print(f"ERROR: Expansion BN in block '{block_path}' has no weight (affine=False?). Cannot determine channel importance. Cannot prune.")
         return model

    n_keep = get_num_channels_to_keep(original_expanded_channels, prune_ratio)
    if n_keep == original_expanded_channels:
        if verbose: print(f"Skipping block '{block_path}': n_keep ({n_keep}) equals original expanded channels ({original_expanded_channels}).")
        return model # No pruning needed for this block

    importance_scores = torch.abs(expand_bn.weight.data)
    sort_idx = torch.argsort(importance_scores, descending=True)
    kept_indices = sort_idx[:n_keep].sort()[0] # Indices of channels to keep

    if verbose: print(f"Pruning block '{block_path}': Expanded channels {original_expanded_channels} -> {n_keep}")

    # Store original and new layers for replacement
    layers_to_replace = {} # {original_object: new_object}

    # 5. Create New Pruned Layers
    bn_device = expand_bn.weight.device
    kept_indices_dev = kept_indices.to(bn_device)

    # Expansion Conv (prune output)
    print("  Creating new expand_conv...")
    new_expand_conv = nn.Conv2d(expand_conv.in_channels, n_keep, 1, 1, 0, bias=(expand_conv.bias is not None))
    if expand_conv.weight is not None:
        new_expand_conv.weight.data.copy_(torch.index_select(expand_conv.weight.detach(), 0, kept_indices_dev))
    if expand_conv.bias is not None:
        new_expand_conv.bias.data.copy_(torch.index_select(expand_conv.bias.detach(), 0, kept_indices_dev))
    layers_to_replace[expand_conv] = new_expand_conv

    # Expansion BN (prune features)
    print("  Creating new expand_bn...")
    new_expand_bn = nn.BatchNorm2d(n_keep, eps=expand_bn.eps, momentum=expand_bn.momentum, affine=expand_bn.affine, track_running_stats=expand_bn.track_running_stats)
    if expand_bn.affine:
        if expand_bn.weight is not None: new_expand_bn.weight.data.copy_(torch.index_select(expand_bn.weight.detach(), 0, kept_indices_dev))
        if expand_bn.bias is not None: new_expand_bn.bias.data.copy_(torch.index_select(expand_bn.bias.detach(), 0, kept_indices_dev))
    if expand_bn.track_running_stats:
        if expand_bn.running_mean is not None: new_expand_bn.running_mean.data.copy_(torch.index_select(expand_bn.running_mean.detach(), 0, kept_indices_dev))
        if expand_bn.running_var is not None: new_expand_bn.running_var.data.copy_(torch.index_select(expand_bn.running_var.detach(), 0, kept_indices_dev))
        if hasattr(expand_bn, 'num_batches_tracked'): new_expand_bn.num_batches_tracked.data.copy_(expand_bn.num_batches_tracked.data)
    layers_to_replace[expand_bn] = new_expand_bn

    # Depthwise Conv (prune input, output, groups)
    print("  Creating new dw_conv...")
    new_dw_conv = nn.Conv2d(n_keep, n_keep, dw_conv.kernel_size, dw_conv.stride, dw_conv.padding, groups=n_keep, bias=(dw_conv.bias is not None), dilation=dw_conv.dilation)
    if dw_conv.weight is not None:
        # Select groups/output channels to keep
        new_dw_conv.weight.data.copy_(torch.index_select(dw_conv.weight.detach(), 0, kept_indices.to(dw_conv.weight.device)))
    if dw_conv.bias is not None:
        new_dw_conv.bias.data.copy_(torch.index_select(dw_conv.bias.detach(), 0, kept_indices.to(dw_conv.bias.device)))
    layers_to_replace[dw_conv] = new_dw_conv

    # Depthwise BN (prune features)
    print("  Creating new dw_bn...")
    new_dw_bn = nn.BatchNorm2d(n_keep, eps=dw_bn.eps, momentum=dw_bn.momentum, affine=dw_bn.affine, track_running_stats=dw_bn.track_running_stats)
    # Copy parameters using kept_indices
    if dw_bn.affine:
        if dw_bn.weight is not None: new_dw_bn.weight.data.copy_(torch.index_select(dw_bn.weight.detach(), 0, kept_indices_dev))
        if dw_bn.bias is not None: new_dw_bn.bias.data.copy_(torch.index_select(dw_bn.bias.detach(), 0, kept_indices_dev))
    if dw_bn.track_running_stats:
        if dw_bn.running_mean is not None: new_dw_bn.running_mean.data.copy_(torch.index_select(dw_bn.running_mean.detach(), 0, kept_indices_dev))
        if dw_bn.running_var is not None: new_dw_bn.running_var.data.copy_(torch.index_select(dw_bn.running_var.detach(), 0, kept_indices_dev))
        if hasattr(dw_bn, 'num_batches_tracked'): new_dw_bn.num_batches_tracked.data.copy_(dw_bn.num_batches_tracked.data)
    layers_to_replace[dw_bn] = new_dw_bn

    # SE Block (prune reduce input, expand output)
    if has_se:
        print("  Creating new SE layers...")
        # Reduce
        bottleneck_ch = reduce_layer.out_channels # Assume bottleneck size doesn't change
        new_reduce = nn.Conv2d(n_keep, bottleneck_ch, 1, 1, 0, bias=(reduce_layer.bias is not None))
        if reduce_layer.weight is not None:
            new_reduce.weight.data.copy_(torch.index_select(reduce_layer.weight.detach(), 1, kept_indices.to(reduce_layer.weight.device))) # Prune input dim
        if reduce_layer.bias is not None: new_reduce.bias.data.copy_(reduce_layer.bias.data) # Bias depends on output channels
        layers_to_replace[reduce_layer] = new_reduce
        # Expand
        new_expand = nn.Conv2d(bottleneck_ch, n_keep, 1, 1, 0, bias=(expand_layer.bias is not None))
        if expand_layer.weight is not None:
            new_expand.weight.data.copy_(torch.index_select(expand_layer.weight.detach(), 0, kept_indices.to(expand_layer.weight.device))) # Prune output dim
        if expand_layer.bias is not None:
            new_expand.bias.data.copy_(torch.index_select(expand_layer.bias.detach(), 0, kept_indices.to(expand_layer.bias.device)))
        layers_to_replace[expand_layer] = new_expand

    # Pointwise Conv (prune input)
    print("  Creating new point_linear_conv...")
    new_point_conv = nn.Conv2d(n_keep, point_conv.out_channels, 1, 1, 0, bias=(point_conv.bias is not None), groups=point_conv.groups, dilation=point_conv.dilation) # Keep original output channels
    if point_conv.weight is not None:
        new_point_conv.weight.data.copy_(torch.index_select(point_conv.weight.detach(), 1, kept_indices.to(point_conv.weight.device))) # Prune input dim
    if point_conv.bias is not None: new_point_conv.bias.data.copy_(point_conv.bias.data) # Bias depends on output channels
    layers_to_replace[point_conv] = new_point_conv

    # 6. Replace Layers within the specific block
    print(f"Replacing layers within block '{block_path}'...")
    replacement_count = 0
    for old_layer, new_layer in layers_to_replace.items():
        # Use replace_layer starting search from the block itself for efficiency
        if replace_layer(block_to_prune, old_layer, new_layer):
            replacement_count += 1
        else:
            # Fallback: Search from the root model if not found in block (shouldn't happen ideally)
            print(f"  Warning: Failed to replace layer {type(old_layer)} directly in block '{block_path}'. Trying search from root model.")
            if replace_layer(model, old_layer, new_layer):
                 replacement_count += 1
            else:
                 print(f"  ERROR: Failed to replace layer {type(old_layer)} (originally from block '{block_path}') even from root.")

    print(f"Block pruning complete for '{block_path}'. Replaced {replacement_count} layers.")
    return model




def run_uniform_pruning_experiments(
    model_base,
    test_loader,
    pruning_ratios,
    base_acc,
    base_flops,
    SEClass,
    input_shape=(1, 3, 96, 96),
    classifier_layer='classifier.linear.weight',
    prune_classifier=True
):
    results = []

    for ratio in pruning_ratios:
        print(f"\n Pruning ratio: {ratio}")

        # Clone, sort, and then prune model
        model = copy.deepcopy(model_base)
        # model = apply_conv2d_sorting(model, device)
        pruned_model = uniform_channel_prune(
            model=model,
            prune_ratio=ratio,
            SqueezeExcitationModuleType=SEClass,
            verbose=False
        )

        # prune final FC layer (e.g., classifier)
        if prune_classifier:
            sparsity_config = {classifier_layer: ratio}
            pruner = Pruner(pruned_model, sparsity_config, mode='col-based')
            pruner.apply(pruned_model)
            pruned_model = col_based_prune_reduction(pruned_model, [classifier_layer])

        pruned_model.to(device)

        # Evaluate
        top1_acc = evaluate(pruned_model, test_loader, device)
        model_size = get_model_size(pruned_model)
        flops = count_net_flops(pruned_model, input_shape)

        # Baseline (assume known or set at first run)

        acc_drop = base_acc - top1_acc
        flops_reduction = 100 * (base_flops - flops) / base_flops

        results.append({
            'Pruning Ratio': ratio,
            'Top-1 Accuracy (%)': round(top1_acc, 2),
            'Accuracy Drop (%)': round(acc_drop, 2),
            'Model Size (MB)': round(model_size, 2),
            'FLOPs (Millions)': round(flops / 1e6, 2),
            'FLOPs Reduction (%)': round(flops_reduction, 2),
        })

    return pd.DataFrame(results)





import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


def run_distillation(teacher_model, student_model, train_loader, num_epochs, device): 
    # --- Configuration ---
    TEMPERATURE = 4.0
    ALPHA = 0.3
    LEARNING_RATE_DISTILL = 1e-3
    # LEARNING_RATE_FINETUNE = 1e-4 # Not used in this snippet
    # BATCH_SIZE = 32 # train_loader has its own batch_size
    # DO_FINETUNING = True # Not used in this snippet
    DEVICE = device # Use the passed device

    # Load models
    teacher_model.eval()
    teacher_model.to(DEVICE) # Ensure teacher is on the correct device
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model_copy = copy.deepcopy(student_model)
    student_model_copy.to(DEVICE) # Ensure student is on the correct device
    
    print("\n--- Starting Knowledge Distillation ---")
    criterion_hard = nn.CrossEntropyLoss()
    # For KLDivLoss, student output should be log_softmax and teacher output should be softmax
    criterion_soft = nn.KLDivLoss(reduction='batchmean', log_target=False) 
    optimizer_student = optim.Adam(student_model_copy.parameters(), lr=LEARNING_RATE_DISTILL)

    # --- Distillation Training ---
    for epoch in range(num_epochs):
        student_model_copy.train()
        running_loss, running_hard_loss, running_soft_loss = 0.0, 0.0, 0.0
        
        progress_bar_desc = f"Distill Epoch {epoch+1}/{num_epochs}"


        progress_bar = tqdm(train_loader, desc=progress_bar_desc)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer_student.zero_grad()

            # Teacher provides soft targets
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # Student predictions
            student_logits = student_model_copy(inputs)

            # Soft loss (Distillation loss)
            # Teacher outputs are softened
            teacher_soft_targets = F.softmax(teacher_logits / TEMPERATURE, dim=1)
            # Student outputs are log_softmax for KLDivLoss
            student_log_soft_outputs = F.log_softmax(student_logits / TEMPERATURE, dim=1)
            
            distill_loss = criterion_soft(student_log_soft_outputs, teacher_soft_targets) * (TEMPERATURE ** 2)
            
            # Hard loss (Standard cross-entropy with true labels)
            hard_loss = criterion_hard(student_logits, labels)
            
            combined_loss = ALPHA * hard_loss + (1 - ALPHA) * distill_loss
            
            combined_loss.backward()
            optimizer_student.step()

            running_loss += combined_loss.item() * inputs.size(0)
            running_hard_loss += hard_loss.item() * inputs.size(0)
            running_soft_loss += distill_loss.item() * inputs.size(0)
            
            progress_bar.set_postfix({
                'Loss': combined_loss.item(),
                'HardL': hard_loss.item(),
                'SoftL': distill_loss.item()
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_hard_loss = running_hard_loss / len(train_loader.dataset)
        epoch_soft_loss = running_soft_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Distill Stats: Avg Loss: {epoch_loss:.4f}, Avg HardL: {epoch_hard_loss:.4f}, Avg SoftL: {epoch_soft_loss:.4f}")

    print("Distillation training complete.")
    return student_model_copy