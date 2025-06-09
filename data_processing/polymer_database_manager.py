# =========================================================================
# Simple Usage Example for Polymer Database Manager
# =========================================================================
# Place polymer_database_manager.py in /content/Inverse_copolymer_design/data_processing/
# Then use this workflow in your Colab notebook

import importlib
import sys
import os
import pandas as pd
import shutil

# Make sure Python can find the script
sys.path.append('/content/Inverse_copolymer_design')

# Import and reload your script to pick up edits
from data_processing import polymer_database_manager
importlib.reload(polymer_database_manager)

# Import the manager
from data_processing.polymer_database_manager import PolymerDatabaseManager

print("✓ Polymer Database Manager imported successfully!")

# ========================
# EXAMPLE 1: Basic Usage
# ========================

# Your CSV file path
input_path = '/content/drive/MyDrive/AI_MSE_Company/Inverse_Design/Case_0/band_gap_chain.csv'

# Check if file exists
if os.path.exists(input_path):
    print(f"✓ Found file: {input_path}")
    
    # Get just the filename
    selected_file = os.path.basename(input_path)
    print(f"Selected file: {selected_file}")
    
    # Create local data directory and copy file
    os.makedirs('data', exist_ok=True)
    local_input = f'data/{selected_file}'
    shutil.copy2(input_path, local_input)
    
    print(f"✓ Copied to: {local_input}")
else:
    print(f"✗ File not found: {input_path}")

# ========================
# Initialize Manager
# ========================

# For new database (no existing template)
manager = PolymerDatabaseManager(verbose=True)

# For existing template (if you have one)
# manager = PolymerDatabaseManager('your_existing_template.csv', verbose=True)

# ========================
# Process Your Dataset
# ========================

print("Starting data processing...")
print("=" * 60)

# Method 1: Interactive mode (recommended for first time)
# This will let you select which columns to use as targets
processed_df = manager.process_new_dataset(
    input_path=local_input,
    expand_variants=True,        # Creates all polymer type/composition variants
    generate_iupac=True,         # Generates IUPAC names (placeholder)
    interactive=True             # Interactive column selection
)

# Method 2: Non-interactive mode (auto-detect all numeric columns)
# processed_df = manager.process_new_dataset(
#     input_path=local_input,
#     expand_variants=True,
#     interactive=False           # Auto-detect all numeric columns
# )

# Method 3: Specify exact columns you want (no interaction needed)
# processed_df = manager.process_new_dataset(
#     input_path=local_input,
#     expand_variants=True,
#     interactive=False,
#     target_columns=['band_gap', 'other_property'],  # Your actual column names
#     column_mapping={'band_gap': 'Band_Gap_eV', 'other_property': 'Other_Property'}
# )

print(f"\n✓ Processed dataset shape: {processed_df.shape}")
print(f"✓ Columns: {list(processed_df.columns)}")

# ========================
# Save Results
# ========================

# Set output file name
output_file = f'data/processed_{selected_file.replace(".csv", "_polymer_database.csv")}'

# Save the processed dataset
combined_df = manager.append_to_template(processed_df, output_file)

print(f"\n✓ Saved to: {output_file}")
print(f"✓ Final database shape: {combined_df.shape}")

# ========================
# Show Results
# ========================

print("\nSample of processed data:")
print("-" * 60)

# Show the structure
sample_cols = ['poly_id', 'poly_type', 'comp', 'fracA', 'fracB', 'monoA', 'monoB']
available_cols = [col for col in sample_cols if col in processed_df.columns]
print(processed_df[available_cols].head())

print("\nSample ChemProp inputs:")
print("-" * 30)
for i in range(min(3, len(processed_df))):
    print(f"\nExample {i+1}:")
    print(f"Master: {processed_df.iloc[i]['master_chemprop_input']}")
    print(f"Poly:   {processed_df.iloc[i]['poly_chemprop_input']}")

# Show target properties
target_cols = [col for col in processed_df.columns 
               if col not in ['poly_id', 'poly_type', 'comp', 'fracA', 'fracB', 'monoA', 'monoB', 
                             'monoA_IUPAC', 'monoB_IUPAC', 'master_chemprop_input', 'poly_chemprop_input']]
if target_cols:
    print(f"\nTarget properties found: {target_cols}")
    for col in target_cols:
        if processed_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            print(f"{col}: {processed_df[col].min():.3f} to {processed_df[col].max():.3f}")

print("\n🎉 Processing complete!")

# ========================
# EXAMPLE 2: Adding More Data to Existing Database
# ========================

print("\n" + "="*60)
print("EXAMPLE 2: Adding New Data to Existing Database")
print("="*60)

# If you have another dataset to add
# new_input_path = '/path/to/your/new/dataset.csv'

# Use the existing database as template
# manager_with_template = PolymerDatabaseManager(output_file, verbose=True)

# Process the new dataset
# new_processed_df = manager_with_template.process_new_dataset(
#     input_path=new_input_path,
#     expand_variants=True,
#     interactive=True  # or False if you want to auto-detect
# )

# Append to existing database
# final_combined_df = manager_with_template.append_to_template(
#     new_processed_df, 
#     'data/final_polymer_database.csv'
# )

# print(f"✓ Final combined database shape: {final_combined_df.shape}")

# ========================
# EXAMPLE 3: Custom Polymer Types/Compositions
# ========================

print("\n" + "="*60)
print("EXAMPLE 3: Custom Polymer Configurations")
print("="*60)

# You can customize the polymer types and compositions
custom_manager = PolymerDatabaseManager(verbose=True)

# Override default configurations
custom_manager.default_poly_types = ['alternating', 'block', 'random', 'gradient']
custom_manager.default_compositions = ['4A_4B', '6A_2B', '2A_6B', '8A_2B']
custom_manager.comp_fracs.update({
    '8A_2B': (0.8, 0.2)  # Add new composition
})

print("✓ Custom configurations set")
print(f"Polymer types: {custom_manager.default_poly_types}")
print(f"Compositions: {custom_manager.default_compositions}")

# ========================
# UTILITY FUNCTIONS
# ========================

def quick_upload_and_process():
    """Helper function for uploading files in Colab"""
    from google.colab import files
    
    print("Please upload your CSV file:")
    uploaded = files.upload()
    
    filename = list(uploaded.keys())[0]
    print(f"✓ Uploaded: {filename}")
    
    # Process the file
    manager = PolymerDatabaseManager(verbose=True)
    processed_df = manager.process_new_dataset(
        input_path=filename,
        expand_variants=True,
        interactive=True
    )
    
    # Save results
    output_name = f"processed_{filename}"
    combined_df = manager.append_to_template(processed_df, output_name)
    
    print(f"✓ Processing complete! Output saved as: {output_name}")
    
    # Offer to download
    download = input("Download the processed file? (y/n): ").strip().lower()
    if download == 'y':
        files.download(output_name)
    
    return combined_df

def reload_manager():
    """Helper function to reload the module during development"""
    import importlib
    importlib.reload(polymer_database_manager)
    from data_processing.polymer_database_manager import PolymerDatabaseManager
    print("✓ Module reloaded")
    return PolymerDatabaseManager

# ========================
# SUMMARY
# ========================

print("\n" + "="*60)
print("USAGE SUMMARY")
print("="*60)

print("""
✓ polymer_database_manager.py placed in data_processing folder
✓ No hardcoded target properties - works with ANY numeric columns
✓ Interactive column selection when needed
✓ Automatic polymer variant expansion
✓ Chemical validation and error handling
✓ Template preservation and database growth

Key Features:
• Handles both homopolymers and copolymers
• Generates master_chemprop_input and poly_chemprop_input
• Preserves ALL your target properties (no matter what they are)
• Interactive column selection and renaming
• Robust chemical validation
• Flexible polymer type and composition configuration

Ready for your polymer datasets! 🧪
""")
