import torch
import argparse
from collections import OrderedDict

def list_model_state_dict(ckpt_path):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    
    # Extract the state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Create an ordered dictionary to store layer names and shapes
    layers = OrderedDict()
    
    # Iterate through the state_dict and store layer names and shapes
    for key, value in state_dict.items():
        layers[key] = value.shape
    
    # Print the layers
    print("Model State Dict:")
    for layer_name, shape in layers.items():
        print(f"{layer_name}: {shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List model state_dict from a checkpoint file")
    parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()
    
    list_model_state_dict(args.ckpt_path)
