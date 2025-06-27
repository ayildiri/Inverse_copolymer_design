# data_processing/transfer_data_utils.py
import pandas as pd
import torch
import numpy as np
import os

def validate_data_vocab_consistency(dict_loader, vocab, dataset_name="dataset"):
    """
    CRITICAL: Validate that all token IDs in the data are within vocabulary bounds
    """
    print(f"🔍 Validating {dataset_name} consistency with vocabulary...")
    
    max_token_found = -1
    total_sequences = 0
    problematic_batches = []
    
    for batch_key in dict_loader:
        try:
            batch_data = dict_loader[batch_key][0]
            if hasattr(batch_data, 'tgt_token_ids'):
                for seq_idx, token_sequence in enumerate(batch_data.tgt_token_ids):
                    total_sequences += 1
                    
                    # Convert to list if it's a tensor
                    if isinstance(token_sequence, torch.Tensor):
                        token_list = token_sequence.tolist()
                    else:
                        token_list = list(token_sequence)
                    
                    # Check each token ID
                    for token_id in token_list:
                        if isinstance(token_id, (int, float)) and token_id >= 0:
                            max_token_found = max(max_token_found, int(token_id))
                            
                            # Critical check: token ID must be within vocab bounds
                            if int(token_id) >= len(vocab):
                                problematic_batches.append({
                                    'batch': batch_key,
                                    'sequence': seq_idx,
                                    'token_id': int(token_id),
                                    'vocab_size': len(vocab)
                                })
                                
                                # Stop at first problem for quick debugging
                                if len(problematic_batches) >= 5:
                                    break
                    
                    if len(problematic_batches) >= 5:
                        break
            
            if len(problematic_batches) >= 5:
                break
                
        except Exception as e:
            print(f"⚠️ Error processing batch {batch_key}: {e}")
            continue
    
    # Report results
    if problematic_batches:
        print(f"❌ CRITICAL ERROR: Found {len(problematic_batches)} token ID mismatches!")
        print(f"   Vocabulary size: {len(vocab)}")
        print(f"   Max token ID found: {max_token_found}")
        print(f"   Problematic examples:")
        for prob in problematic_batches[:3]:
            print(f"     Batch {prob['batch']}, seq {prob['sequence']}: token_id={prob['token_id']} >= vocab_size={prob['vocab_size']}")
        
        print(f"\n🔧 SOLUTION:")
        print(f"   1. Regenerate data files with correct vocabulary")
        print(f"   2. Or use a vocabulary file that matches the data")
        print(f"   3. Check that tokenization method matches between training and inference")
        
        raise ValueError(f"Token IDs in {dataset_name} exceed vocabulary size. Max token: {max_token_found}, Vocab size: {len(vocab)}")
    
    else:
        print(f"✅ {dataset_name} validation passed!")
        print(f"   Checked {total_sequences} sequences")
        print(f"   Max token ID: {max_token_found} (within vocab size {len(vocab)})")
        print(f"   Vocabulary coverage: {(max_token_found + 1) / len(vocab) * 100:.1f}%")

def load_transfer_data(csv_path, stage, source_properties, target_properties, 
                      batch_size, tokenization, vocab, sample_weight=1.0, 
                      device='cuda', dataset_path=None):
    """
    Transfer learning data loading that works with stage-specific directories
    ENHANCED with vocabulary validation
    """
    
    if dataset_path is None:
        # Default paths based on stage
        base_path = "/content/drive/MyDrive/X_Materials_Organized_Files_V1/2_Inverse_Design/Transfer_Learning"
        if stage == 1:
            dataset_path = os.path.join(base_path, "Stage1_Data")
        else:
            dataset_path = os.path.join(base_path, "Stage2_Data")
    
    print(f"📂 Loading data from: {dataset_path}")
    print(f"🎯 Stage: {stage}")
    print(f"🔤 Tokenization: {tokenization}")
    print(f"📚 Vocabulary size: {len(vocab)}")
    
    if stage == 1:
        print(f"🧬 Stage 1: Loading dataset with {source_properties}")
        
        # Files have been standardized without property suffixes
        train_loader_path = os.path.join(dataset_path, f'dict_train_loader_augmented_{tokenization}.pt')
        val_loader_path = os.path.join(dataset_path, f'dict_val_loader_augmented_{tokenization}.pt')
        test_loader_path = os.path.join(dataset_path, f'dict_test_loader_augmented_{tokenization}.pt')
        
        # Check if files exist
        if not os.path.exists(train_loader_path):
            print(f"❌ ERROR: Could not find {train_loader_path}")
            print(f"📁 Available files in {dataset_path}:")
            if os.path.exists(dataset_path):
                for f in sorted(os.listdir(dataset_path)):
                    if f.endswith('.pt'):
                        print(f"  - {f}")
            else:
                print(f"❌ Directory {dataset_path} does not exist!")
            raise FileNotFoundError(f"Stage 1 data files not found in {dataset_path}")
        
        print(f"📥 Loading data files...")
        try:
            dict_train_loader = torch.load(train_loader_path, map_location='cpu')  # Load to CPU first
            dict_val_loader = torch.load(val_loader_path, map_location='cpu')
            dict_test_loader = torch.load(test_loader_path, map_location='cpu')
            print(f"✅ Successfully loaded data files")
        except Exception as e:
            print(f"❌ Error loading data files: {e}")
            raise
        
        # CRITICAL: Validate vocabulary consistency IMMEDIATELY after loading
        print(f"\n🔍 PERFORMING CRITICAL VOCABULARY VALIDATION...")
        try:
            validate_data_vocab_consistency(dict_train_loader, vocab, "training data")
            validate_data_vocab_consistency(dict_val_loader, vocab, "validation data")
            validate_data_vocab_consistency(dict_test_loader, vocab, "test data")
            print(f"✅ All vocabulary validations passed!")
        except Exception as e:
            print(f"❌ VOCABULARY VALIDATION FAILED: {e}")
            print(f"\n🔧 This is likely the cause of your embedding dimension mismatch!")
            raise
        
        print(f"📊 Stage 1: Loaded {len(dict_train_loader)} training batches with properties {source_properties}")
        
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
        
        # Apply weighted sampling if requested
        if sample_weight != 1.0 and unlabeled_count > 0:
            print(f"\n   🎯 Applying weighted sampling with weight={sample_weight} for labeled data")
            dict_train_loader = apply_weighted_sampling(dict_train_loader, sample_weight)
            
            # Recount after weighted sampling
            new_labeled, new_unlabeled = check_data_composition(dict_train_loader, 'y1', 'y2')
            new_total = new_labeled + new_unlabeled
            print(f"   📊 After weighted sampling:")
            print(f"      Labeled: {new_labeled:,} ({new_labeled/new_total*100:.1f}%)")
            print(f"      Unlabeled: {new_unlabeled:,} ({new_unlabeled/new_total*100:.1f}%)")
            print(f"      Total batches: {len(dict_train_loader)}")
        
        # Check validation and test set composition too
        val_labeled, val_unlabeled = check_data_composition(dict_val_loader, 'y1', 'y2')
        test_labeled, test_unlabeled = check_data_composition(dict_test_loader, 'y1', 'y2')
        
        print(f"\n   📊 Validation set: {val_labeled:,} labeled, {val_unlabeled:,} unlabeled")
        print(f"   📊 Test set: {test_labeled:,} labeled, {test_unlabeled:,} unlabeled")
        
    else:  # Stage 2
        print(f"🧬 Stage 2: Loading dataset with {target_properties}")
        
        # For Stage 2, files are named with bandgap pattern
        # Files have been standardized without property suffixes
        train_loader_path = os.path.join(dataset_path, f'dict_train_loader_augmented_{tokenization}.pt')
        val_loader_path = os.path.join(dataset_path, f'dict_val_loader_augmented_{tokenization}.pt')
        test_loader_path = os.path.join(dataset_path, f'dict_test_loader_augmented_{tokenization}.pt')
                
        # Check if files exist
        if not os.path.exists(train_loader_path):
            print(f"❌ ERROR: Could not find {train_loader_path}")
            print(f"📁 Available files in {dataset_path}:")
            if os.path.exists(dataset_path):
                for f in sorted(os.listdir(dataset_path)):
                    if f.endswith('.pt'):
                        print(f"  - {f}")
            else:
                print(f"❌ Directory {dataset_path} does not exist!")
            raise FileNotFoundError(f"Stage 2 data files not found in {dataset_path}")
        
        print(f"📥 Loading data files...")
        try:
            dict_train_loader = torch.load(train_loader_path, map_location='cpu')
            dict_val_loader = torch.load(val_loader_path, map_location='cpu')
            dict_test_loader = torch.load(test_loader_path, map_location='cpu')
            print(f"✅ Successfully loaded data files")
        except Exception as e:
            print(f"❌ Error loading data files: {e}")
            raise
        
        # CRITICAL: Validate vocabulary consistency for Stage 2 as well
        print(f"\n🔍 PERFORMING CRITICAL VOCABULARY VALIDATION...")
        try:
            validate_data_vocab_consistency(dict_train_loader, vocab, "Stage 2 training data")
            validate_data_vocab_consistency(dict_val_loader, vocab, "Stage 2 validation data")
            validate_data_vocab_consistency(dict_test_loader, vocab, "Stage 2 test data")
            print(f"✅ All Stage 2 vocabulary validations passed!")
        except Exception as e:
            print(f"❌ STAGE 2 VOCABULARY VALIDATION FAILED: {e}")
            raise
        
        print(f"📊 Stage 2: Loaded {len(dict_train_loader)} training batches with property {target_properties}")
        
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
    
    # Move data to correct device if specified
    if device != 'cpu':
        print(f"📱 Moving data to device: {device}")
        dict_train_loader = move_data_to_device(dict_train_loader, device)
        dict_val_loader = move_data_to_device(dict_val_loader, device)
        dict_test_loader = move_data_to_device(dict_test_loader, device)
    
    print(f"✅ Data loading completed successfully!")
    return dict_train_loader, dict_val_loader, dict_test_loader

def move_data_to_device(dict_loader, device):
    """
    Move all tensors in data loader to specified device
    """
    try:
        for batch_key in dict_loader:
            # Move the graph data
            dict_loader[batch_key][0].to(device)
            # Move the matrices
            dict_loader[batch_key][1].to(device)
            dict_loader[batch_key][2].to(device)
        return dict_loader
    except Exception as e:
        print(f"⚠️ Warning: Could not move data to device {device}: {e}")
        return dict_loader

def verify_tokenization_consistency(dict_loader, tokenization, vocab):
    """
    Verify that the data was created with the expected tokenization method
    """
    print(f"🔍 Verifying tokenization consistency...")
    
    # Check a few sample sequences
    sample_count = 0
    for batch_key in list(dict_loader.keys())[:3]:  # Check first 3 batches
        batch_data = dict_loader[batch_key][0]
        if hasattr(batch_data, 'tgt_token_ids'):
            for seq in batch_data.tgt_token_ids[:2]:  # Check first 2 sequences per batch
                sample_count += 1
                
                # Convert token IDs back to tokens for inspection
                if isinstance(seq, torch.Tensor):
                    token_ids = seq.tolist()
                else:
                    token_ids = list(seq)
                
                # Basic checks for tokenization patterns
                # RT_tokenized should have specific patterns
                if tokenization == "RT_tokenized":
                    # Look for polymer-specific tokens
                    has_polymer_tokens = any(vocab.get(tid, '') in ['|', '[*:', ':', '<', '>'] 
                                           for tid in token_ids if tid < len(vocab))
                    if not has_polymer_tokens and sample_count < 5:
                        print(f"⚠️ Warning: Sample {sample_count} doesn't show expected RT_tokenized patterns")
                
                if sample_count >= 5:
                    break
        if sample_count >= 5:
            break
    
    print(f"✅ Checked {sample_count} sample sequences for tokenization consistency")

def apply_weighted_sampling(dict_loader, sample_weight):
    """
    Apply weighted sampling to oversample batches with more labeled data.
    This duplicates batches that have high labeled content.
    """
    print(f"🎯 Applying weighted sampling with weight={sample_weight}...")
    
    # First, analyze each batch
    batch_info = []
    for batch_key in dict_loader:
        batch_data = dict_loader[batch_key][0]
        labeled_count = 0
        
        for i in range(batch_data.num_graphs):
            if hasattr(batch_data, 'y1') and hasattr(batch_data, 'y2'):
                y1_val = batch_data.y1[i] if i < len(batch_data.y1) else float('nan')
                y2_val = batch_data.y2[i] if i < len(batch_data.y2) else float('nan')
                
                if not torch.isnan(y1_val) and not torch.isnan(y2_val):
                    labeled_count += 1
        
        labeled_ratio = labeled_count / batch_data.num_graphs
        batch_info.append((batch_key, labeled_ratio, labeled_count))
    
    # Create new dict with weighted sampling
    new_dict_loader = {}
    new_batch_idx = 0
    
    for batch_key, labeled_ratio, labeled_count in batch_info:
        # Always include the original batch
        new_dict_loader[str(new_batch_idx)] = dict_loader[batch_key]
        new_batch_idx += 1
        
        # Duplicate batches with high labeled content
        if labeled_ratio > 0.5:  # If more than 50% labeled
            # Number of duplications based on sample_weight
            n_duplicates = int((sample_weight - 1) * labeled_ratio)
            for _ in range(n_duplicates):
                new_dict_loader[str(new_batch_idx)] = dict_loader[batch_key]
                new_batch_idx += 1
        elif labeled_ratio > 0:  # Batches with some labeled data
            # Probabilistically duplicate based on labeled ratio and sample_weight
            if np.random.random() < (sample_weight - 1) * labeled_ratio:
                new_dict_loader[str(new_batch_idx)] = dict_loader[batch_key]
                new_batch_idx += 1
    
    print(f"   📈 Expanded from {len(dict_loader)} to {len(new_dict_loader)} batches")
    
    return new_dict_loader

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
                print(f"📝 Note: Created dummy y2 for batches with only y1")
            modified_count += 1
    
    if modified_count > 0:
        print(f"   ✏️ Modified {modified_count} batches to have y2 property")
    
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
