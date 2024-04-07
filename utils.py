import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import os


def generate_same_padding(kernel_size: int):
    padding = int((kernel_size-1)/2)
    return padding


def generate_same_padding_transpose(kernel_size: int,
                                    stride: int):
    padding = int((kernel_size-stride+1)//2)
    return padding


def num_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def view_batch(batch,cols):
    """To view a batch of generated images during training."""
    assert len(batch.shape) == 4
    size = batch.shape[0]
    assert size%cols==0
    for i in range(size):
        plt.subplot(size//cols, cols, i+1)
        plt.axis("off")
        plt.imshow(batch[i])
    plt.show()


def sort_celebs_into_sets(path,set_size):
    """Distributes dataset into different directories for separate training."""
    
    _list = os.listdir(path=path)
    last_index=-1
    
    for i in tqdm(range(len(_list)//set_size)):
        new_dir  = os.path.join(path, f"set_{i}")

        if not os.path.exists(new_dir):  # Check if the directory already exists
            try:
                os.mkdir(new_dir)  # Create a new destination directory if it doesn't exist
            except OSError as e:
                print(f"Error creating directory '{new_dir}': {str(e)}")
                continue  # Skip to the next directory if there's an error

        for j in range(set_size):
            try:
                # Move the file to the new directory
                shutil.move(f"{path}/{_list[i*set_size+j]}", f"{new_dir}/{_list[i*set_size+j]}")
            except FileNotFoundError:
                print(f"File '{_list[i*set_size+j]}' not found.")
            except shutil.Error as e:
                print(f"Error moving file '{_list[i*set_size+j]}' to '{new_dir}': {str(e)}")
                
        last_index=i
        
    #Handles any remaining files.
    if len(_list)//set_size != 0:
        print(f"Sorting final {len(_list)-set_size*last_index+1} images into final directory.")
        destination_directory = os.path.join(path, f"set_{last_index + 1}")
        
        if not os.path.exists(destination_directory):
            try:
                os.mkdir(destination_directory)  # Create a new destination directory if it doesn't exist
            except OSError as e:
                print(f"Error creating directory '{destination_directory}': {str(e)}")
                return
        
        for i in tqdm(range(len(_list)-set_size*last_index+1)):
            try:
                # Move the remaining file to the final destination directory
                shutil.move(f"{path}/{_list[last_index*set_size+1+i]}", f"{new_dir}/{_list[last_index*set_size+1+i]}")
            except FileNotFoundError:
                print(f"File '{_list[last_index*set_size+1+i]}' not found.")
            except shutil.Error as e:
                print(f"Error moving file '{_list[last_index * set_size + 1 + i]}' to '{destination_directory}': {str(e)}")