import pandas as pd
import torch
import numpy as np
import os

def load_transfer_data(csv_path, stage, source_properties, target_properties, 
                      batch_size, tokenization, vocab, sample_weight=1.0, 
                      device='cuda', dataset_path='/content/renamed_dataset'):
    """
    Simplified transfer learning data loading that works with your existing structure
    """
    
    if stage == 1:
        # Stage 1: Need to load the original paper's dataset with EA/IP
        print(f"Stage 1: Loading original dataset with EA/IP properties")
        
        # Try to load the original (non-augmented) dataset that should have EA/IP
        train_loader_path = os.path.join(dataset_path, f'dict_train_loader_original_{tokenization}.pt')
        val_loader_path = os.path.join(dataset_path, f'dict_val_loader_original_{tokenization}.pt')
        test_loader_path = os.path.join(dataset_path, f'dict_test_loader_original_{tokenization}.pt')
        
        # If original dataset doesn't exist, we need to create it from the paper's data
        if not os.path.exists(train_loader_path):
            print("WARNING: Original dataset with EA/IP not found.")
            print("For now, using augmented dataset. You'll need to prepare the EA/IP dataset.")
            
            # Fallback to augmented dataset
            train_loader_path = os.path.join(dataset_path, f'dict_train_loader_augmented_{tokenization}.pt')
            val_loader_path = os.path.join(dataset_path, f'dict_val_loader_augmented_{tokenization}.pt')
            test_loader_path = os.path.join(dataset_path, f'dict_test_loader_augmented_{tokenization}.pt')
        
        dict_train_loader = torch.load(train_loader_path)
        dict_val_loader = torch.load(val_loader_path)
        dict_test_loader = torch.load(test_loader_path)
        
        # For Stage 1 with EA/IP, we need y1=EA, y2=IP
        # If using augmented data as fallback, create dummy y2
        dict_train_loader = prepare_stage1_properties(dict_train_loader, device)
        dict_val_loader = prepare_stage1_properties(dict_val_loader, device)
        dict_test_loader = prepare_stage1_properties(dict_test_loader, device)
        
    else:  # Stage 2
        print(f"Stage 2: Loading augmented dataset with bandgap")
        
        # Load your current augmented dataset
        train_loader_path = os.path.join(dataset_path, f'dict_train_loader_augmented_{tokenization}.pt')
        val_loader_path = os.path.join(dataset_path, f'dict_val_loader_augmented_{tokenization}.pt')
        test_loader_path = os.path.join(dataset_path, f'dict_test_loader_augmented_{tokenization}.pt')
        
        dict_train_loader = torch.load(train_loader_path)
        dict_val_loader = torch.load(val_loader_path)
        dict_test_loader = torch.load(test_loader_path)
        
        # For Stage 2, y1 should be bandgap (which it already is)
        print(f"Stage 2: Loaded {len(dict_train_loader)} training batches")
    
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
