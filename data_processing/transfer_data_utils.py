# data_processing/transfer_data_utils.py
import pandas as pd
import torch
import numpy as np
import os

def load_transfer_data(csv_path, stage, source_properties, target_properties, 
                      batch_size, tokenization, vocab, sample_weight=1.0, 
                      device='cuda', dataset_path=None):
    """
    Transfer learning data loading that works with stage-specific directories
    """
    
    # ADD THIS BLOCK HERE (between lines 11-12)
    if dataset_path is None:
        # Default paths based on stage
        base_path = "/content/drive/MyDrive/X_Materials_Organized_Files_V1/2_Inverse_Design/Transfer_Learning"
        if stage == 1:
            dataset_path = os.path.join(base_path, "Stage1_Data")
        else:
            dataset_path = os.path.join(base_path, "Stage2_Data")
    
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
        
        # For semi-supervised Stage 1, check data composition
        labeled_count = 0
        unlabeled_count = 0
        
        for batch_key in dict_train_loader:
            batch_data = dict_train_loader[batch_key][0]
            for i in range(batch_data.num_graphs):
                if hasattr(batch_data, 'y1') and hasattr(batch_data, 'y2'):
                    # Check if both EA and IP are non-NaN
                    y1_val = batch_data.y1[i] if i < len(batch_data.y1) else float('nan')
                    y2_val = batch_data.y2[i] if i < len(batch_data.y2) else float('nan')
                    
                    if not torch.isnan(y1_val) and not torch.isnan(y2_val):
                        labeled_count += 1
                    else:
                        unlabeled_count += 1
                else:
                    # If properties don't exist, count as unlabeled
                    unlabeled_count += batch_data.num_graphs
                    break
        
        print(f"   📊 Training data composition:")
        print(f"      Labeled (EA+IP): {labeled_count:,}")
        print(f"      Unlabeled: {unlabeled_count:,}")
        print(f"      Total: {labeled_count + unlabeled_count:,}")
        
        if unlabeled_count > 0:
            print(f"   ✅ Semi-supervised learning enabled!")
            print(f"   📈 Labeled ratio: {labeled_count / (labeled_count + unlabeled_count):.1%}")
        else:
            print(f"   ⚠️  No unlabeled data found - running supervised learning only")
        
        # Check validation and test set composition too
        val_labeled, val_unlabeled = check_data_composition(dict_val_loader, 'y1', 'y2')
        test_labeled, test_unlabeled = check_data_composition(dict_test_loader, 'y1', 'y2')
        
        print(f"\n   📊 Validation set: {val_labeled:,} labeled, {val_unlabeled:,} unlabeled")
        print(f"   📊 Test set: {test_labeled:,} labeled, {test_unlabeled:,} unlabeled")
        
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
        
        # Check Stage 2 data composition
        labeled_count = 0
        unlabeled_count = 0
        
        for batch_key in dict_train_loader:
            batch_data = dict_train_loader[batch_key][0]
            for i in range(batch_data.num_graphs):
                if hasattr(batch_data, 'y1'):
                    y1_val = batch_data.y1[i] if i < len(batch_data.y1) else float('nan')
                    if not torch.isnan(y1_val):
                        labeled_count += 1
                    else:
                        unlabeled_count += 1
                else:
                    unlabeled_count += batch_data.num_graphs
                    break
        
        print(f"   📊 Training data composition:")
        print(f"      Labeled (bandgap): {labeled_count:,}")
        print(f"      Unlabeled: {unlabeled_count:,}")
        print(f"      Total: {labeled_count + unlabeled_count:,}")
        
        if unlabeled_count > 0:
            print(f"   ⚠️  Warning: Stage 2 contains unlabeled data - this is unusual")
    
    return dict_train_loader, dict_val_loader, dict_test_loader

def check_data_composition(dict_loader, prop1='y1', prop2=None):
    """
    Check how many labeled vs unlabeled molecules are in a data loader
    """
    labeled_count = 0
    unlabeled_count = 0
    
    for batch_key in dict_loader:
        batch_data = dict_loader[batch_key][0]
        for i in range(batch_data.num_graphs):
            is_labeled = False
            
            # Check first property
            if hasattr(batch_data, prop1):
                val1 = getattr(batch_data, prop1)[i] if i < len(getattr(batch_data, prop1)) else float('nan')
                if not torch.isnan(val1):
                    is_labeled = True
                    
                    # If second property specified, check it too
                    if prop2 and hasattr(batch_data, prop2):
                        val2 = getattr(batch_data, prop2)[i] if i < len(getattr(batch_data, prop2)) else float('nan')
                        if torch.isnan(val2):
                            is_labeled = False
            
            if is_labeled:
                labeled_count += 1
            else:
                unlabeled_count += 1
    
    return labeled_count, unlabeled_count

def prepare_stage1_properties(dict_loader, device):
    """
    Prepare properties for Stage 1 training
    If we only have y1 (bandgap), create dummy y2 for now
    """
    modified_count = 0
    
    for batch_key in dict_loader:
        batch_data = dict_loader[batch_key][0]
        
        # Check what properties exist
        if hasattr(batch_data, 'y1') and not hasattr(batch_data, 'y2'):
            # If only y1 exists, create dummy y2 with NaN values
            batch_size = batch_data.y1.shape[0]
            batch_data.y2 = torch.full((batch_size,), float('nan'), device=batch_data.y1.device)
            if modified_count == 0:  # Only print once
                print(f"Note: Created dummy y2 for batches with only y1")
            modified_count += 1
    
    if modified_count > 0:
        print(f"   Modified {modified_count} batches to have y2 property")
    
    return dict_loader

def analyze_batch_properties(batch_data, batch_idx=0):
    """
    Analyze property distribution in a single batch for debugging
    """
    print(f"\n🔍 Analyzing batch {batch_idx}:")
    print(f"   Num graphs: {batch_data.num_graphs}")
    
    if hasattr(batch_data, 'y1'):
        y1_valid = sum(~torch.isnan(batch_data.y1))
        print(f"   y1 (EA) valid values: {y1_valid}/{len(batch_data.y1)}")
    else:
        print(f"   y1 (EA): NOT PRESENT")
    
    if hasattr(batch_data, 'y2'):
        y2_valid = sum(~torch.isnan(batch_data.y2))
        print(f"   y2 (IP) valid values: {y2_valid}/{len(batch_data.y2)}")
    else:
        print(f"   y2 (IP): NOT PRESENT")
    
    if hasattr(batch_data, 'y3'):
        y3_valid = sum(~torch.isnan(batch_data.y3))
        print(f"   y3 (bandgap) valid values: {y3_valid}/{len(batch_data.y3)}")
    else:
        print(f"   y3 (bandgap): NOT PRESENT")

def verify_semi_supervised_setup(dict_train_loader, dict_val_loader, dict_test_loader, stage=1):
    """
    Verify that semi-supervised learning is properly set up
    """
    print(f"\n🔍 VERIFYING SEMI-SUPERVISED SETUP FOR STAGE {stage}")
    print("="*60)
    
    # Analyze first batch in detail
    if '0' in dict_train_loader:
        first_batch = dict_train_loader['0'][0]
        analyze_batch_properties(first_batch, batch_idx=0)
    
    # Summary statistics
    train_labeled, train_unlabeled = check_data_composition(dict_train_loader, 'y1', 'y2' if stage == 1 else None)
    val_labeled, val_unlabeled = check_data_composition(dict_val_loader, 'y1', 'y2' if stage == 1 else None)
    test_labeled, test_unlabeled = check_data_composition(dict_test_loader, 'y1', 'y2' if stage == 1 else None)
    
    total_molecules = (train_labeled + train_unlabeled + val_labeled + val_unlabeled + test_labeled + test_unlabeled)
    total_labeled = train_labeled + val_labeled + test_labeled
    total_unlabeled = train_unlabeled + val_unlabeled + test_unlabeled
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total molecules: {total_molecules:,}")
    print(f"   Total labeled: {total_labeled:,} ({total_labeled/total_molecules*100:.1f}%)")
    print(f"   Total unlabeled: {total_unlabeled:,} ({total_unlabeled/total_molecules*100:.1f}%)")
    
    if stage == 1 and total_unlabeled > 0:
        print(f"\n✅ Semi-supervised learning is properly configured!")
        print(f"   The model will learn polymer grammar from ALL {total_molecules:,} molecules")
        print(f"   Property prediction will use only the {total_labeled:,} labeled molecules")
    elif stage == 1:
        print(f"\n⚠️  No unlabeled data found - Stage 1 will run as supervised learning only")
    
    return {
        'train': {'labeled': train_labeled, 'unlabeled': train_unlabeled},
        'val': {'labeled': val_labeled, 'unlabeled': val_unlabeled},
        'test': {'labeled': test_labeled, 'unlabeled': test_unlabeled},
        'total': {'labeled': total_labeled, 'unlabeled': total_unlabeled, 'all': total_molecules}
    }
