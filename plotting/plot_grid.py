from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import Draw
from math import ceil
import argparse
import os


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

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties (for output file naming)")
parser.add_argument("--input_file", type=str, default=None,
                    help="Input file path (if not specified, uses default pattern)")
parser.add_argument("--scaling_factor", type=float, default=1.0,
                    help="Scaling factor for file naming")
parser.add_argument("--grid_size", type=int, default=11,
                    help="Size of the grid (grid_size x grid_size)")
parser.add_argument("--output_prefix", type=str, default="seed_literature_grid",
                    help="Output file prefix")
parser.add_argument("--results_path", type=str, default=".",
                    help="Path to results directory")
parser.add_argument("--font_size", type=int, default=10,
                    help="Font size for overlay text")
parser.add_argument("--figure_size", type=int, default=15,
                    help="Figure size in inches")

args = parser.parse_args()

# Define polymer classification labels
classes_stoich = [['0.5','0.5'],['0.25','0.75'],['0.75','0.25']]
classes_con = ['<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25','<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5','<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.375:0.375<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125']

labels_stoich = {'0.5|0.5':'1:1','0.25|0.75':'1:3','0.75|0.25':'3:1'}
labels_con = {'0.5':'A','0.25':'R','0.375':'B'}

smiles_list = []
stoich_con_list = []

# Determine input file path
if args.input_file:
    file_path = args.input_file
else:
    # Use default pattern with scaling factor
    file_path = os.path.join(args.results_path, f'seed_literature_grid_scaling{args.scaling_factor}.txt')

print(f"Reading from: {file_path}")

# Check if file exists
if not os.path.exists(file_path):
    print(f"Error: Input file {file_path} not found!")
    exit(1)

# Read and parse the input file
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line_num, line in enumerate(lines, 1):
        # Skip header lines and empty lines
        if line.startswith("Seed molecule:") or line.startswith("The following are the generations from seed"):
            continue
        elif line.strip():
            try:
                # Parse polymer string
                smiles = line.split("|")[0]
                print(f"Line {line_num}: {line.strip()}")
                
                # Extract connectivity and stoichiometry
                con = line.split("|")[-1].split(':')[1]
                stoich = "|".join(line.split("|")[1:3])
                
                # Create labels
                stoich_con_label = "".join([labels_stoich[stoich], ' ', labels_con[con]])
                stoich_con_list.append(stoich_con_label)
                smiles_list.append(smiles)
                
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not parse line {line_num}: {line.strip()}")
                print(f"Error: {e}")
                continue

print(f"Successfully parsed {len(smiles_list)} molecules")

# Calculate required grid size
total_molecules = len(smiles_list)
required_grid_size = ceil(total_molecules**0.5)

# Use provided grid size or calculated minimum
grid_size = max(args.grid_size, required_grid_size)
total_grid_spots = grid_size * grid_size

print(f"Using {grid_size}x{grid_size} grid for {total_molecules} molecules")

# Convert SMILES to images
placeholder_image = create_placeholder_image()
molecule_images = [smiles_to_image(smiles) for smiles in smiles_list]

# Check for conversion failures
valid_images = sum(1 for img in molecule_images if img is not None)
print(f"Successfully converted {valid_images}/{len(smiles_list)} SMILES to images")

# Create grids for images and labels
image_grid = np.empty((grid_size, grid_size), dtype=object)
inf_grid = np.empty((grid_size, grid_size), dtype=object)

# Fill the grid
idx = 0
for i in range(grid_size):
    for j in range(grid_size):
        if idx < len(molecule_images) and molecule_images[idx]:
            image_grid[i, j] = molecule_images[idx]
            inf_grid[i, j] = stoich_con_list[idx] if idx < len(stoich_con_list) else ""
        else:
            image_grid[i, j] = placeholder_image
            inf_grid[i, j] = "Empty" if idx >= len(molecule_images) else "Invalid"
        idx += 1

# Create the plot
fig, axes = plt.subplots(grid_size, grid_size, figsize=(args.figure_size, args.figure_size))

# Handle the case where grid_size is 1
if grid_size == 1:
    axes = [[axes]]
elif grid_size > 1:
    # Ensure axes is always 2D
    if len(axes.shape) == 1:
        axes = axes.reshape(-1, 1)

# Plot each image in the grid
for i in range(grid_size):
    for j in range(grid_size):
        ax = axes[i][j] if grid_size > 1 else axes[i][j]
        ax.imshow(image_grid[i, j])
        ax.axis('off')
        
        # Add text overlay with stoichiometry and connectivity labels
        if inf_grid[i, j]:
            ax.text(0.5, 0.5, inf_grid[i, j], ha='center', va='center', 
                   transform=ax.transAxes, fontsize=args.font_size, 
                   color='black', alpha=0.3, weight="bold")

# Generate output filename with property-aware naming for consistency
property_suffix = "_".join(args.property_names) if len(args.property_names) <= 3 else f"{len(args.property_names)}props"
output_filename = f'{args.output_prefix}_scaling{args.scaling_factor}_{property_suffix}.png'
output_path = os.path.join(args.results_path, output_filename)

# Save the plot
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
print(f"Grid plot saved to: {output_path}")

# Display summary
print(f"\nSummary:")
print(f"Grid size: {grid_size}x{grid_size}")
print(f"Total molecules: {total_molecules}")
print(f"Valid images: {valid_images}")
print(f"Properties context: {args.property_names}")
print(f"Scaling factor: {args.scaling_factor}")

plt.close(fig)
