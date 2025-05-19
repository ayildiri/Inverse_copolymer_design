import sys, os
main_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir_path)
import pandas as pd
from statistics import mean
import numpy as np
import argparse
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


parser = argparse.ArgumentParser()
parser.add_argument("--augment", help="options: augmented, original", default="augmented", choices=["augmented", "original"])
parser.add_argument("--tokenization", help="options: oldtok, RT_tokenized", default="oldtok", choices=["oldtok", "RT_tokenized"])
parser.add_argument("--embedding_dim", help="latent dimension (equals word embedding dimension in this model)", default=32)
parser.add_argument("--beta", default=1, help="option: <any number>, schedule", choices=["normalVAE","schedule"])
parser.add_argument("--loss", default="ce", choices=["ce","wce"])
parser.add_argument("--AE_Warmup", default=False, action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initialization", default="random", choices=["random"])
parser.add_argument("--add_latent", type=int, default=1)
parser.add_argument("--ppguided", type=int, default=0)
parser.add_argument("--dec_layers", type=int, default=4)
parser.add_argument("--max_beta", type=float, default=0.1)
parser.add_argument("--max_alpha", type=float, default=0.1)
parser.add_argument("--epsilon", type=float, default=1)

# Add flexible property arguments
parser.add_argument("--property_names", type=str, nargs='+', default=["EA", "IP"],
                    help="Names of the properties to evaluate")
parser.add_argument("--property_count", type=int, default=None,
                    help="Number of properties (auto-detected from property_names if not specified)")
parser.add_argument("--property_units", type=str, nargs='+', default=None,
                    help="Units for each property (e.g., 'eV'). If not specified, uses 'eV' for all.")

args = parser.parse_args()

# Handle property configuration
property_names = args.property_names
if args.property_count is not None:
    property_count = args.property_count
else:
    property_count = len(property_names)

# Validate that property count matches property names
if len(property_names) != property_count:
    raise ValueError(f"Number of property names ({len(property_names)}) must match property count ({property_count})")

# Handle property units
if args.property_units:
    if len(args.property_units) != property_count:
        raise ValueError(f"Number of property units ({len(args.property_units)}) must match property count ({property_count})")
    property_units = args.property_units
else:
    property_units = ["eV"] * property_count

print(f"Evaluating predictions for {property_count} properties: {property_names}")
print(f"Property units: {property_units}")

seed = args.seed
augment = args.augment #augmented or original
tokenization = args.tokenization #oldtok or RT_tokenized
if args.add_latent ==1:
    add_latent=True
elif args.add_latent ==0:
    add_latent=False

dataset_type = "train"
data_augment = "old"

# Include property info in model name
property_str = "_".join(property_names) if len(property_names) <= 3 else f"{len(property_names)}props"
model_name = 'Model_'+data_augment+'data_DecL='+str(args.dec_layers)+'_beta='+str(args.beta)+'_maxbeta='+str(args.max_beta)+'_maxalpha='+str(args.max_alpha)+'eps='+str(args.epsilon)+'_loss='+str(args.loss)+'_augment='+str(args.augment)+'_tokenization='+str(args.tokenization)+'_AE_warmup='+str(args.AE_Warmup)+'_init='+str(args.initialization)+'_seed='+str(args.seed)+'_add_latent='+str(add_latent)+'_pp-guided='+str(args.ppguided)+'_props='+property_str+'/'

# Directory to save results
dir_name = os.path.join(main_dir_path,'Checkpoints/', model_name)
if not os.path.exists(dir_name):
    print(f"Error: Model directory does not exist: {dir_name}")
    exit(1)

# Load property data dynamically
property_data_real = {}
property_data_pred = {}

# Load real property values (y1_all, y2_all, ...)
for i in range(property_count):
    y_file = os.path.join(dir_name, f'y{i+1}_all_{dataset_type}.npy')
    if os.path.exists(y_file):
        with open(y_file, 'rb') as f:
            property_data_real[f'y{i+1}'] = list(np.load(f))
        print(f"Loaded {property_names[i]} real values from {y_file}")
    else:
        print(f"Warning: Real property file not found: {y_file}")
        property_data_real[f'y{i+1}'] = []

# Load predicted property values (yp_all)
yp_file = os.path.join(dir_name, f'yp_all_{dataset_type}.npy')
if os.path.exists(yp_file):
    with open(yp_file, 'rb') as f:
        yp_all = np.load(f)
    print(f"Loaded predicted values from {yp_file}")
    
    # Extract individual property predictions
    for i in range(property_count):
        if yp_all.shape[1] > i:
            property_data_pred[f'yp{i+1}'] = [yp[i] for yp in yp_all]
        else:
            print(f"Warning: Not enough properties in predicted data for {property_names[i]}")
            property_data_pred[f'yp{i+1}'] = []
else:
    print(f"Error: Predicted property file not found: {yp_file}")
    exit(1)

# Check data consistency
for i in range(property_count):
    real_key = f'y{i+1}'
    pred_key = f'yp{i+1}'
    if real_key in property_data_real and pred_key in property_data_pred:
        real_len = len(property_data_real[real_key])
        pred_len = len(property_data_pred[pred_key])
        print(f"{property_names[i]}: {pred_len} predictions, {real_len} real values")

def calculate_rmse_r2(real, predicted):
    """Calculate RMSE and R2 scores, handling NaN values."""
    # Remove NaNs from real and corresponding values from predicted
    real_values = [r for r, p in zip(real, predicted) if not np.isnan(r)]
    predicted_values = [p for r, p in zip(real, predicted) if not np.isnan(r)]
    
    if len(real_values) == 0:
        return np.nan, np.nan
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(real_values, predicted_values))
    
    # Calculate R-squared
    r2 = r2_score(real_values, predicted_values)
    
    return rmse, r2

def create_parity_plot(real_data, pred_data, property_name, property_unit, rmse, r2, save_path):
    """Create a parity plot for a given property."""
    # Filter out NaN values
    real = [r for r, p in zip(real_data, pred_data) if not np.isnan(r)]
    predicted = [p for r, p in zip(real_data, pred_data) if not np.isnan(r)]
    
    if len(real) == 0:
        print(f"Warning: No valid data for {property_name}, skipping parity plot")
        return
    
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
    
    plt.scatter(real, predicted, color='blue', s=0.1)
    plt.plot(real, real, color='black', linestyle='--')
    plt.xlabel(f'Real Values ({property_unit})')
    plt.ylabel(f'Predicted Values ({property_unit})')
    plt.title(f'{property_name}')
    plt.grid(True)
    
    # Add RMSE and R2 text
    textstr = f'RMSE: {rmse:.3f}\nR²: {r2:.3f}'
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parity plot: {save_path}")

def create_kde_plot(real_data, pred_data, property_name, property_unit, save_path):
    """Create a KDE plot comparing real and predicted distributions."""
    # Filter data: real distribution (values with real labels), augmented distribution (predicted values where real is NaN)
    real_distribution = np.array([r for r, p in zip(real_data, pred_data) if not np.isnan(r)])
    augmented_distribution = np.array([p for r, p in zip(real_data, pred_data) if np.isnan(r)])
    
    if len(real_distribution) == 0 and len(augmented_distribution) == 0:
        print(f"Warning: No valid data for {property_name}, skipping KDE plot")
        return
    
    # Reshape the data
    if len(real_distribution) > 0:
        real_distribution = real_distribution.reshape(-1, 1)
    if len(augmented_distribution) > 0:
        augmented_distribution = augmented_distribution.reshape(-1, 1)
    
    plt.figure(figsize=(10, 6))
    
    # Define bandwidth
    bandwidth = 0.1
    
    # Create combined data range for x-axis
    all_values = []
    if len(real_distribution) > 0:
        all_values.extend(real_distribution.flatten())
    if len(augmented_distribution) > 0:
        all_values.extend(augmented_distribution.flatten())
    
    if len(all_values) == 0:
        print(f"Warning: No data to plot for {property_name}")
        return
    
    x_values = np.linspace(min(all_values), max(all_values), 1000)
    
    # Plot real data distribution if available
    if len(real_distribution) > 0:
        kde_real = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_real.fit(real_distribution)
        real_density = np.exp(kde_real.score_samples(x_values.reshape(-1, 1)))
        plt.plot(x_values, real_density, label='Real Data')
    
    # Plot augmented data distribution if available
    if len(augmented_distribution) > 0:
        kde_augmented = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde_augmented.fit(augmented_distribution)
        augmented_density = np.exp(kde_augmented.score_samples(x_values.reshape(-1, 1)))
        plt.plot(x_values, augmented_density, label='Augmented Data')
    
    plt.xlabel(f'{property_name} ({property_unit})')
    plt.ylabel('Density')
    plt.title(f'Kernel Density Estimation ({property_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved KDE plot: {save_path}")

# Calculate metrics and create plots for each property
metrics_summary = {}

print("\n" + "="*60)
print("PROPERTY PREDICTION EVALUATION RESULTS")
print("="*60)

for i in range(property_count):
    property_name = property_names[i]
    property_unit = property_units[i]
    real_key = f'y{i+1}'
    pred_key = f'yp{i+1}'
    
    print(f"\n{property_name}:")
    print("-" * 40)
    
    if real_key in property_data_real and pred_key in property_data_pred:
        real_data = property_data_real[real_key]
        pred_data = property_data_pred[pred_key]
        
        # Calculate metrics
        rmse, r2 = calculate_rmse_r2(real_data, pred_data)
        metrics_summary[property_name] = {'RMSE': rmse, 'R2': r2}
        
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Create parity plot
        parity_save_path = os.path.join(dir_name, f'parity_{property_name}.png')
        create_parity_plot(real_data, pred_data, property_name, property_unit, rmse, r2, parity_save_path)
        
        # Create KDE plot
        kde_save_path = os.path.join(dir_name, f'KDE_{property_name}.png')
        create_kde_plot(real_data, pred_data, property_name, property_unit, kde_save_path)
    else:
        print(f"Error: Missing data for {property_name}")

# Save metrics summary
summary_file = os.path.join(dir_name, 'evaluation_metrics_summary.txt')
with open(summary_file, 'w') as f:
    f.write("Property Prediction Evaluation Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Properties evaluated: {property_names}\n")
    f.write(f"Property units: {property_units}\n")
    f.write(f"Dataset: {dataset_type}\n\n")
    
    f.write("Metrics:\n")
    f.write("-"*30 + "\n")
    for prop_name, metrics in metrics_summary.items():
        f.write(f"{prop_name}:\n")
        f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"  R²: {metrics['R2']:.4f}\n\n")

print(f"\n✅ Evaluation completed!")
print(f"📊 Summary saved to: {summary_file}")
print(f"📈 Plots saved to: {dir_name}")
print("\nMetrics Summary:")
for prop_name, metrics in metrics_summary.items():
    print(f"  {prop_name}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
