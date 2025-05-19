from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw
from math import ceil
import ast
import argparse
import json
import os
import re
import glob


# Function to create a placeholder image with Matplotlib
def create_placeholder_image(size=(200, 200), text="Invalid"):
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax.text(0.5, 0.5, text, fontsize=20, ha='center', va='center')
    ax.axis('off')
    fig.canvas.draw()
    
    # Convert the plot to an image
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image

# Function to convert SMILES to image or placeholder if invalid
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol)
    else:
        return None

def parse_filename_for_properties(filename):
    """Extract property information from filename if available."""
    # Try to extract property info from filename pattern like:
    # top20_mols_GA_correct_EAmin_props=EA_IP_iter_1000_run1.txt
    # top20_mols_GA_correct_EAmin_props=ThermalCond_MechStrength_iter_1000_run1.txt
    
    # Pattern with property names
    match_with_props = re.search(r"top20_mols_GA_correct_([A-Za-z0-9]+)_props=([A-Za-z0-9_]+)_(time|iter)_(\d+)_run(\d+)", filename)
    if match_with_props:
        objective_type = match_with_props.group(1)
        property_names = match_with_props.group(2).split('_')
        time_or_iter = match_with_props.group(3)
        time_iter_value = match_with_props.group(4)
        run_number = match_with_props.group(5)
        return objective_type, property_names, time_or_iter, time_iter_value, run_number
    
    # Legacy pattern without explicit property names - assume EA/IP
    match_legacy = re.search(r"top20_mols_GA_correct_([A-Za-z0-9]+)_(time|iter)_(\d+)_run(\d+)", filename)
    if match_legacy:
        objective_type = match_legacy.group(1)
        property_names = ["EA", "IP"]  # Default for backward compatibility
        time_or_iter = match_legacy.group(2)
        time_iter_value = match_legacy.group(3)
        run_number = match_legacy.group(4)
        return objective_type, property_names, time_or_iter, time_iter_value, run_number
    
    return None, None, None, None, None

def calculate_legacy_objective(val_EA, val_IP, objective_type):
    """Calculate objective value for legacy EA/IP objectives."""
    if objective_type == 'mimick_peak':
        return abs(val_EA + 2.0) + abs(val_IP - 1.2)
    elif objective_type == 'mimick_best':
        return abs(val_EA + 2.64) + abs(val_IP - 1.61)
    elif objective_type == 'EAmin':
        return val_EA + abs(val_IP - 1.0)
    elif objective_type == 'max_gap':
        return val_EA - val_IP
    else:
        # For unknown objectives, just return the sum
        return val_EA + val_IP

def format_property_display(property_names, property_values):
    """Format property values for display."""
    lines = []
    for i, (name, value) in enumerate(zip(property_names, property_values)):
        if name in ["EA", "IP"]:
            # Add units for known properties
            lines.append(f"{name} (eV): {value:.3f}")
        else:
            # Use generic formatting for other properties
            lines.append(f"{name}: {value:.3f}")
    return "\n".join(lines)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties")
parser.add_argument("--results_path", type=str, default="plotting/results/",
                    help="Path to results directory")
parser.add_argument("--file_pattern", type=str, default="top20_mols_GA_correct_*.txt",
                    help="File pattern to search for")
parser.add_argument("--grid_x", type=int, default=2, help="Grid size in x direction")
parser.add_argument("--grid_y", type=int, default=5, help="Grid size in y direction")
parser.add_argument("--output_prefix", type=str, default="optimal_mols_GA_correct",
                    help="Output file prefix")

args = parser.parse_args()

classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']

labels_stoich = {'0.5|0.5':'1:1','0.25|0.75':'1:3','0.75|0.25':'3:1'}
labels_con = {'0.5':'A','0.25':'R','0.375':'B'}

smiles_list = []
stoich_con_list = []
obj_val_list = []
property_lists = [[] for _ in args.property_names]  # Dynamic list for each property
poly_strings = []

scaling_factor = 1.0

# Search for files
search_path = os.path.join(args.results_path, args.file_pattern)

# Loop through each file matching the pattern
for filepath in glob.glob(search_path):
    filename = os.path.basename(filepath)
    
    # Extract information from filename
    objective_type, file_property_names, time_or_iter, time_iter_value, run_number = parse_filename_for_properties(filename)
    
    if not objective_type:
        print(f"Warning: Could not parse filename {filename}, skipping...")
        continue
    
    # Use property names from file if available, otherwise use command line args
    if file_property_names:
        property_names = file_property_names
    else:
        property_names = args.property_names
    
    print(f"Processing file: {filename}")
    print(f"Objective: {objective_type}, Properties: {property_names}")
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        mols_dict = ast.literal_eval(lines[0])
        line2 = re.sub(r'tensor\(([^)]+)\)', r'\1', lines[1])
        props_dict = ast.literal_eval(line2)
        
        for iteration, poly_string in mols_dict.items():
            if not poly_string in poly_strings:
                smiles = poly_string.split("|")[0]
                con = poly_string.split("|")[-1].split(':')[1]
                stoich = "|".join(poly_string.split("|")[1:3])
                stoich_con_list.append("".join([labels_stoich[stoich], ' ', labels_con[con]]))
                smiles_list.append(smiles)
                
                # Extract property values dynamically
                property_values = []
                try:
                    prop_data = props_dict[iteration]
                    for i in range(len(property_names)):
                        if i < len(prop_data):
                            property_values.append(float(prop_data[i]))
                            property_lists[i].append(float(prop_data[i]))
                        else:
                            # Handle missing properties
                            property_values.append(0.0)
                            property_lists[i].append(0.0)
                    
                    # Calculate objective value
                    if objective_type in ['mimick_peak', 'mimick_best', 'EAmin', 'max_gap'] and len(property_values) >= 2:
                        # Legacy EA/IP objectives
                        val_obj = calculate_legacy_objective(property_values[0], property_values[1], objective_type)
                    else:
                        # For custom objectives or unknown types, use sum of absolute values
                        val_obj = sum(abs(val) for val in property_values)
                    
                    obj_val_list.append(round(val_obj, 3))
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error processing properties for iteration {iteration}: {e}")
                    # Add default values
                    for i in range(len(property_names)):
                        property_lists[i].append(0.0)
                    obj_val_list.append(0.0)
                
                poly_strings.append(poly_string)

# Ensure we have at least some molecules to plot
if not smiles_list:
    print("No valid molecules found to plot!")
    exit(1)

# Convert SMILES to images
grid_size_x = args.grid_x
grid_size_y = args.grid_y
max_molecules = grid_size_x * grid_size_y
placeholder_image = create_placeholder_image()

molecule_images = [smiles_to_image(smiles) for smiles in smiles_list[:max_molecules]]
print(f"Found {len(molecule_images)} molecules to plot")

# Create grids
image_grid = np.empty((grid_size_x, grid_size_y), dtype=object)
inf_grid = np.empty((grid_size_x, grid_size_y), dtype=object)
inf_grid_obj = np.empty((grid_size_x, grid_size_y), dtype=object)

# Place molecule images in the grid
idx = 0
for i in range(grid_size_x):
    for j in range(grid_size_y):
        if idx < len(molecule_images) and molecule_images[idx]:
            image_grid[i, j] = molecule_images[idx]
            inf_grid[i, j] = stoich_con_list[idx] if idx < len(stoich_con_list) else ""
            
            # Format objective and property display
            obj_str = f"f(z)={obj_val_list[idx]:.3f}\n" if idx < len(obj_val_list) else "f(z)=N/A\n"
            
            # Add property values dynamically
            if idx < len(property_lists[0]):
                property_values = [prop_list[idx] for prop_list in property_lists]
                property_str = format_property_display(property_names, property_values)
                inf_grid_obj[i, j] = obj_str + property_str
            else:
                inf_grid_obj[i, j] = obj_str + "Properties: N/A"
        else:
            image_grid[i, j] = placeholder_image
            inf_grid[i, j] = ""
            inf_grid_obj[i, j] = "Invalid"
        idx += 1

# Plot the grid
plt.rcParams.update({'font.size': 18, 'font.family': 'sans-serif'})

fig, axes = plt.subplots(grid_size_x, grid_size_y, figsize=(10, 4))
plt.subplots_adjust(hspace=0.6)

# Handle single row/column case
if grid_size_x == 1:
    axes = axes.reshape(1, -1)
elif grid_size_y == 1:
    axes = axes.reshape(-1, 1)

# Plot each image in the grid
for i in range(grid_size_x):
    for j in range(grid_size_y):
        if grid_size_x == 1 and grid_size_y == 1:
            ax = axes
        elif grid_size_x == 1:
            ax = axes[j]
        elif grid_size_y == 1:
            ax = axes[i]
        else:
            ax = axes[i, j]
            
        ax.imshow(image_grid[i, j])
        ax.axis('off')
        
        # Add text overlays
        ax.text(0.5, 0.5, inf_grid[i, j], ha='center', va='center', 
                transform=ax.transAxes, fontsize=11, color='black', alpha=0.3, weight="bold")
        ax.text(0.5, -0.25, inf_grid_obj[i, j], ha='center', va='center', 
                transform=ax.transAxes, fontsize=11, color='black')

# Save with property-aware filename
property_suffix = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
output_filename = f'{args.output_prefix}_{property_suffix}.png'
output_path = os.path.join(args.results_path, output_filename)

plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
print(f"Plot saved to: {output_path}")

# Print summary statistics
if smiles_list:
    print(f"\nSummary:")
    print(f"Total molecules plotted: {len(smiles_list)}")
    print(f"Properties: {property_names}")
    if obj_val_list:
        print(f"Average objective value: {np.mean(obj_val_list):.3f}")
        print(f"Objective values: {obj_val_list}")
    
    # Print property statistics
    for i, prop_name in enumerate(property_names):
        if i < len(property_lists) and property_lists[i]:
            values = property_lists[i]
            print(f"{prop_name} - Mean: {np.mean(values):.3f}, Std: {np.std(values):.3f}, Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
