# data_processing/transfer_data_utils.py
import pandas as pd
import torch
import numpy as np
import os

def validate_data_vocab_consistency(dict_loader, vocab, dataset_name="dataset"):
    """
    CRITICAL: Validate that all token IDs in the data are within vocabulary bounds - FIXED
    """
    print(f"🔍 Validating {dataset_name} consistency with vocabulary...")
    
    max_token_found = -1
    total_sequences = 0
    total_tokens = 0
    problematic_batches = []
    
    for batch_key in dict_loader:
        try:
            batch_data = dict_loader[batch_key][0]
            if hasattr(batch_data, 'tgt_token_ids'):
                for seq_idx, token_sequence in enumerate(batch_data.tgt_token_ids):
                    total_sequences += 1
                    
                    # FIXED: Handle different data types properly
                    if isinstance(token_sequence, torch.Tensor):
                        token_list = token_sequence.detach().cpu().tolist()
                    elif isinstance(token_sequence, np.ndarray):
                        token_list = token_sequence.tolist()
                    elif isinstance(token_sequence, (list, tuple)):
                        token_list = list(token_sequence)
                    else:
                        print(f"⚠️ Unknown token sequence type: {type(token_sequence)}")
                        continue
                    
                    # Check each token ID
                    for token_idx, token_id in enumerate(token_list):
                        total_tokens += 1
                        
                        # FIXED: Handle different token types
                        if isinstance(token_id, torch.Tensor):
                            token_val = token_id.item()
                        elif isinstance(token_id, (int, float, np.integer, np.floating)):
                            token_val = int(token_id)
                        else:
                            print(f"⚠️ Unknown token type: {type(token_id)} = {token_id}")
                            continue
                        
                        # Skip padding and special negative values
                        if token_val >= 0:
                            max_token_found = max(max_token_found, token_val)
                            
                            # Critical check: token ID must be within vocab bounds
                            if token_val >= len(vocab):
                                problematic_batches.append({
                                    'batch': batch_key,
                                    'sequence': seq_idx,
                                    'position': token_idx,
                                    'token_id': token_val,
                                    'vocab_size': len(vocab)
                                })
                                
                                # Stop at first few problems for quick debugging
                                if len(problematic_batches) >= 5:
                                    break
                    
                    if len(problematic_batches) >= 5:
                        break
            else:
                print(f"⚠️ Batch {batch_key} has no tgt_token_ids attribute")
            
            if len(problematic_batches) >= 5:
                break
                
        except Exception as e:
            print(f"⚠️ Error processing batch {batch_key}: {e}")
            continue
    
    # Report results
    print(f"   📊 Checked {total_sequences:,} sequences with {total_tokens:,} total tokens")
    
    if problematic_batches:
        print(f"❌ CRITICAL ERROR: Found {len(problematic_batches)} token ID mismatches!")
        print(f"   Vocabulary size: {len(vocab)}")
        print(f"   Max token ID found: {max_token_found}")
        print(f"   Problematic examples:")
        for prob in problematic_batches[:3]:
            print(f"     Batch {prob['batch']}, seq {prob['sequence']}, pos {prob['position']}: token_id={prob['token_id']} >= vocab_size={prob['vocab_size']}")
        
        print(f"\n🔧 SOLUTION:")
        print(f"   1. Regenerate data files with correct vocabulary")
        print(f"   2. Or use a vocabulary file that matches the data")
        print(f"   3. Check that tokenization method matches between training and inference")
        
        raise ValueError(f"Token IDs in {dataset_name} exceed vocabulary size. Max token: {max_token_found}, Vocab size: {len(vocab)}")
    
    else:
        print(f"✅ {dataset_name} validation passed!")
        print(f"   Max token ID: {max_token_found} (within vocab size {len(vocab)})")
        if max_token_found >= 0:
            print(f"   Vocabulary coverage: {(max_token_found + 1) / len(vocab) * 100:.1f}%")
        else:
            print(f"   ⚠️ No valid token IDs found - this might indicate a data structure issue")
    
    return max_token_found

def get_property_attribute_mapping(property_names):
    """
    Create a dynamic mapping from property names to data attributes (y1, y2, y3, etc.)
    
    Common mappings:
    - Position 0 (y1): First property
    - Position 1 (y2): Second property  
    - Position 2 (y3): Third property
    - etc.
    
    Returns: dict mapping property name to attribute name
    """
    # Standard property mappings (for backward compatibility)
    standard_mappings = {
        'EA': 'y1',
        'IP': 'y2', 
        'bandgap': 'y3',
        'e_homo': 'y1',
        'e_lumo': 'y2',
        'e_gap': 'y3',
        'homo': 'y1',
        'lumo': 'y2',
        'gap': 'y3'
    }
    
    mapping = {}
    for i, prop_name in enumerate(property_names):
        # Check if there's a standard mapping
        if prop_name in standard_mappings:
            mapping[prop_name] = standard_mappings[prop_name]
        else:
            # Otherwise use positional mapping
            mapping[prop_name] = f'y{i+1}'
    
    return mapping
    
def load_transfer_data(csv_path=None, stage=1, source_properties=None, target_properties=None, 
                      batch_size=32, tokenization="RT_tokenized", vocab=None, sample_weight=1.0, 
                      device='cuda', dataset_path=None):
    """
    Transfer learning data loading that works with stage-specific directories
    ENHANCED with vocabulary validation and flexible property support
    
    Args:
        csv_path: Path to CSV file (optional, for future use)
        stage: 1 for pretraining, 2 for fine-tuning
        source_properties: List of property names used in pretraining
        target_properties: List of property names for fine-tuning
        batch_size: Batch size for data loading
        tokenization: Tokenization method
        vocab: Vocabulary dictionary
        sample_weight: Weight for sampling labeled data in stage 1
        device: Device to load data to
        dataset_path: Custom dataset path
    """
    
    # Default to common properties if not specified
    if source_properties is None:
        source_properties = ['EA', 'IP']
    if target_properties is None:
        target_properties = ['bandgap']
    
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
    
    # Get property mappings
    all_properties = list(set(source_properties + target_properties))
    property_mapping = get_property_attribute_mapping(all_properties)
    
    if stage == 1:
        print(f"🧬 Stage 1: Loading dataset with properties: {source_properties}")
        
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
        
        print(f"📊 Stage 1: Loaded {len(dict_train_loader)} training batches")
        
        # For semi-supervised Stage 1, check data composition
        labeled_count = 0
        unlabeled_count = 0
        
        # Get property attributes for source properties
        source_attrs = [property_mapping.get(prop, f'y{i+1}') for i, prop in enumerate(source_properties)]
        
        for batch_key in dict_train_loader:
            batch_data = dict_train_loader[batch_key][0]
            for i in range(batch_data.num_graphs):
                is_labeled = True
                
                # Check if all source properties are non-NaN
                for prop_attr in source_attrs:
                    if hasattr(batch_data, prop_attr):
                        prop_vals = getattr(batch_data, prop_attr)
                        if i < len(prop_vals):
                            if torch.isnan(prop_vals[i]):
                                is_labeled = False
                                break
                    else:
                        is_labeled = False
                        break
                
                if is_labeled:
                    labeled_count += 1
                else:
                    unlabeled_count += 1
        
        print(f"   📊 Training data composition:")
        print(f"      Labeled (all {source_properties}): {labeled_count:,}")
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
            dict_train_loader = apply_weighted_sampling(dict_train_loader, sample_weight, source_properties, property_mapping)
            
            # Recount after weighted sampling
            new_labeled, new_unlabeled = check_data_composition(dict_train_loader, source_properties, property_mapping)
            new_total = new_labeled + new_unlabeled
            print(f"   📊 After weighted sampling:")
            print(f"      Labeled: {new_labeled:,} ({new_labeled/new_total*100:.1f}%)")
            print(f"      Unlabeled: {new_unlabeled:,} ({new_unlabeled/new_total*100:.1f}%)")
            print(f"      Total batches: {len(dict_train_loader)}")
        
        # Check validation and test set composition too
        val_labeled, val_unlabeled = check_data_composition(dict_val_loader, source_properties, property_mapping)
        test_labeled, test_unlabeled = check_data_composition(dict_test_loader, source_properties, property_mapping)
        
        print(f"\n   📊 Validation set: {val_labeled:,} labeled, {val_unlabeled:,} unlabeled")
        print(f"   📊 Test set: {test_labeled:,} labeled, {test_unlabeled:,} unlabeled")
        
    else:  # Stage 2
        print(f"🧬 Stage 2: Loading dataset with properties: {target_properties}")
        
        # For Stage 2, files are named with standard pattern
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
        
        print(f"📊 Stage 2: Loaded {len(dict_train_loader)} training batches")
        
        # Check Stage 2 data composition
        labeled_count = 0
        unlabeled_count = 0
        
        # Get property attributes for target properties
        target_attrs = [property_mapping.get(prop, f'y{i+1}') for i, prop in enumerate(target_properties)]
        
        for batch_key in dict_train_loader:
            batch_data = dict_train_loader[batch_key][0]
            for i in range(batch_data.num_graphs):
                is_labeled = True
                
                # Check if all target properties are non-NaN
                for prop_attr in target_attrs:
                    if hasattr(batch_data, prop_attr):
                        prop_vals = getattr(batch_data, prop_attr)
                        if i < len(prop_vals):
                            if torch.isnan(prop_vals[i]):
                                is_labeled = False
                                break
                    else:
                        is_labeled = False
                        break
                
                if is_labeled:
                    labeled_count += 1
                else:
                    unlabeled_count += 1
        
        print(f"   📊 Training data composition:")
        print(f"      Labeled ({target_properties}): {labeled_count:,}")
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

def apply_weighted_sampling(dict_loader, sample_weight, properties, property_mapping):
    """
    Apply weighted sampling to oversample batches with more labeled data.
    This duplicates batches that have high labeled content.
    
    Args:
        dict_loader: Data loader dictionary
        sample_weight: Weight for oversampling labeled data
        properties: List of property names to check
        property_mapping: Mapping from property names to attributes
    """
    print(f"🎯 Applying weighted sampling with weight={sample_weight}...")
    
    # First, analyze each batch
    batch_info = []
    for batch_key in dict_loader:
        batch_data = dict_loader[batch_key][0]
        labeled_count = 0
        
        for i in range(batch_data.num_graphs):
            is_labeled = True
            
            # Check if all properties are non-NaN
            for prop in properties:
                prop_attr = property_mapping.get(prop, f'y{properties.index(prop)+1}')
                if hasattr(batch_data, prop_attr):
                    prop_vals = getattr(batch_data, prop_attr)
                    if i < len(prop_vals):
                        if torch.isnan(prop_vals[i]):
                            is_labeled = False
                            break
                else:
                    is_labeled = False
                    break
            
            if is_labeled:
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

def check_data_composition(dict_loader, properties, property_mapping):
    """
    Check how many labeled vs unlabeled molecules are in a data loader
    
    Args:
        dict_loader: Data loader dictionary
        properties: List of property names to check
        property_mapping: Mapping from property names to attributes
    """
    labeled_count = 0
    unlabeled_count = 0
    
    for batch_key in dict_loader:
        batch_data = dict_loader[batch_key][0]
        for i in range(batch_data.num_graphs):
            is_labeled = True
            
            # Check if all properties are non-NaN
            for prop in properties:
                prop_attr = property_mapping.get(prop, f'y{properties.index(prop)+1}')
                if hasattr(batch_data, prop_attr):
                    prop_vals = getattr(batch_data, prop_attr)
                    if i < len(prop_vals):
                        if torch.isnan(prop_vals[i]):
                            is_labeled = False
                            break
                else:
                    is_labeled = False
                    break
            
            if is_labeled:
                labeled_count += 1
            else:
                unlabeled_count += 1
    
    return labeled_count, unlabeled_count

def prepare_stage1_properties(dict_loader, device):
    """
    Prepare properties for Stage 1 training
    If some properties are missing, create dummy values with NaN
    """
    modified_count = 0
    
    for batch_key in dict_loader:
        batch_data = dict_loader[batch_key][0]
        
        # Check what properties exist and add missing ones
        # This ensures compatibility with models expecting certain properties
        max_property_idx = 0
        for attr in dir(batch_data):
            if attr.startswith('y') and attr[1:].isdigit():
                max_property_idx = max(max_property_idx, int(attr[1:]))
        
        # Ensure at least y1 and y2 exist (for backward compatibility)
        if max_property_idx < 2:
            for i in range(max_property_idx + 1, 3):
                attr_name = f'y{i}'
                if not hasattr(batch_data, attr_name):
                    batch_size = batch_data.num_graphs
                    setattr(batch_data, attr_name, torch.full((batch_size,), float('nan'), device=device))
                    if modified_count == 0:  # Only print once
                        print(f"📝 Note: Created dummy {attr_name} for batches with missing properties")
                    modified_count += 1
    
    if modified_count > 0:
        print(f"   ✏️ Modified {modified_count} batches to have consistent properties")
    
    return dict_loader

def analyze_batch_properties(batch_data, batch_idx=0, property_names=None):
    """
    Analyze property distribution in a single batch for debugging
    
    Args:
        batch_data: Batch data object
        batch_idx: Batch index for display
        property_names: Optional list of property names for better display
    """
    print(f"\n🔍 Analyzing batch {batch_idx}:")
    print(f"   Num graphs: {batch_data.num_graphs}")
    
    # Check all y* attributes
    property_attrs = []
    for attr in sorted(dir(batch_data)):
        if attr.startswith('y') and attr[1:].isdigit():
            property_attrs.append(attr)
    
    for i, prop_attr in enumerate(property_attrs):
        if hasattr(batch_data, prop_attr):
            prop_vals = getattr(batch_data, prop_attr)
            valid_count = sum(~torch.isnan(prop_vals))
            
            # Use property name if provided
            if property_names and i < len(property_names):
                prop_display = f"{prop_attr} ({property_names[i]})"
            else:
                prop_display = prop_attr
                
            print(f"   {prop_display} valid values: {valid_count}/{len(prop_vals)}")
        else:
            print(f"   {prop_attr}: NOT PRESENT")

def verify_semi_supervised_setup(dict_train_loader, dict_val_loader, dict_test_loader, stage=1, 
                                 source_properties=None, target_properties=None):
    """
    Verify that semi-supervised learning is properly set up
    
    Args:
        dict_train_loader: Training data loader
        dict_val_loader: Validation data loader
        dict_test_loader: Test data loader
        stage: Training stage (1 or 2)
        source_properties: List of source property names
        target_properties: List of target property names
    """
    print(f"\n🔍 VERIFYING SEMI-SUPERVISED SETUP FOR STAGE {stage}")
    print("="*60)
    
    # Determine which properties to check
    if stage == 1:
        properties_to_check = source_properties or ['EA', 'IP']
    else:
        properties_to_check = target_properties or ['bandgap']
    
    property_mapping = get_property_attribute_mapping(properties_to_check)
    
    # Analyze first batch in detail
    if '0' in dict_train_loader:
        first_batch = dict_train_loader['0'][0]
        analyze_batch_properties(first_batch, batch_idx=0, property_names=properties_to_check)
    
    # Summary statistics
    train_labeled, train_unlabeled = check_data_composition(dict_train_loader, properties_to_check, property_mapping)
    val_labeled, val_unlabeled = check_data_composition(dict_val_loader, properties_to_check, property_mapping)
    test_labeled, test_unlabeled = check_data_composition(dict_test_loader, properties_to_check, property_mapping)
    
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
        print(f"   Properties being learned: {properties_to_check}")
    elif stage == 1:
        print(f"\n⚠️  No unlabeled data found - Stage 1 will run as supervised learning only")
        print(f"   Properties being learned: {properties_to_check}")
    
    return {
        'train': {'labeled': train_labeled, 'unlabeled': train_unlabeled},
        'val': {'labeled': val_labeled, 'unlabeled': val_unlabeled},
        'test': {'labeled': test_labeled, 'unlabeled': test_unlabeled},
        'total': {'labeled': total_labeled, 'unlabeled': total_unlabeled, 'all': total_molecules}
    }
