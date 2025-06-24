# data_processing/transfer_data_utils.py

import pandas as pd
import torch
import numpy as np
import os

def load_transfer_data(csv_path, stage, source_properties, target_properties, 
                      batch_size, tokenization, vocab, sample_weight=1.0, 
                      device='cuda', dataset_path='/content/renamed_dataset'):
    """
    Transfer learning data loading that works with stage-specific directories
    """
    
    if stage == 1:
        print(f"Stage 1: Loading dataset with {source_properties}")
        
        # For Stage 1, files are named with EA_IP pattern
        property_str = "_".join(source_properties)  # "EA_IP"
        
        train_loader_path = os.path.join(dataset_path, f'dict_train_loader_augmented_{tokenization}_{property_str}.pt')
        val_loader_path = os.path.join(dataset_path, f'dict_val_loader_augmented_{tokenization}_{property_str}.pt')
        test_loader_path = os.path.join(dataset_path, f'dict_test_loader_augmented_{tokenization}_{property_str}.pt')
        
        # Check if files exist
        if not os.path.exists(train_loader_path):
            print(f"ERROR: Could not find {train_loader_path}")
            print(f"Available files in {dataset_path}:")
            if os.path.exists(dataset_path):
                for f in os.listdir(dataset_path):
                    if f.endswith('.pt'):
                        print(f"  - {f}")
            raise FileNotFoundError(f"Stage 1 data files not found in {dataset_path}")
        
        dict_train_loader = torch.load(train_loader_path)
        dict_val_loader = torch.load(val_loader_path)
        dict_test_loader = torch.load(test_loader_path)
        
        print(f"Stage 1: Loaded {len(dict_train_loader)} training batches with properties {source_properties}")
        
    else:  # Stage 2
        print(f"Stage 2: Loading dataset with {target_properties}")
        
        # For Stage 2, files are named with bandgap pattern
        property_str = "_".join(target_properties)  # "bandgap"
        
        train_loader_path = os.path.join(dataset_path, f'dict_train_loader_augmented_{tokenization}_{property_str}.pt')
        val_loader_path = os.path.join(dataset_path, f'dict_val_loader_augmented_{tokenization}_{property_str}.pt')
        test_loader_path = os.path.join(dataset_path, f'dict_test_loader_augmented_{tokenization}_{property_str}.pt')
        
        # Check if files exist
        if not os.path.exists(train_loader_path):
            print(f"ERROR: Could not find {train_loader_path}")
            print(f"Available files in {dataset_path}:")
            if os.path.exists(dataset_path):
                for f in os.listdir(dataset_path):
                    if f.endswith('.pt'):
                        print(f"  - {f}")
            raise FileNotFoundError(f"Stage 2 data files not found in {dataset_path}")
        
        dict_train_loader = torch.load(train_loader_path)
        dict_val_loader = torch.load(val_loader_path)
        dict_test_loader = torch.load(test_loader_path)
        
        print(f"Stage 2: Loaded {len(dict_train_loader)} training batches with property {target_properties}")
    
    return dict_train_loader, dict_val_loader, dict_test_loader

def prepare_stage1_properties(dict_loader, device):
    """
    Prepare properties for Stage 1 training
    If we only have y1 (bandgap), create dummy y2 for now
    """
    for batch_key in dict_loader:
        batch_data = dict_loader[batch_key][0]
        
        # Check what properties exist
        if hasattr(batch_data, 'y1') and not hasattr(batch_data, 'y2'):
            # If only y1 exists, create dummy y2 with NaN values
            batch_size = batch_data.y1.shape[0]
            batch_data.y2 = torch.full((batch_size,), float('nan'), device=batch_data.y1.device)
            print(f"Note: Created dummy y2 for batch {batch_key}")
    
    return dict_loader
