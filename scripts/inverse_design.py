import torch
import argparse
from model.G2S_clean import *
from data_processing.data_utils import *

def inverse_design_polymers(model_path, target_properties, conditions=None, num_samples=100):
    """Generate polymers with target properties"""
    
    # Load model
    checkpoint = torch.load(model_path)
    model_config = checkpoint['model_config']
    
    # Initialize model
    model = ModularPolymerVAE(
        base_vae_class=G2S_VAE_PPguided,
        config_path=model_config['property_config_path'],
        **model_config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate polymers
    generated = model.inverse_design(
        target_properties=target_properties,
        conditions=conditions,
        num_samples=num_samples
    )
    
    return generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--target_Tg", type=float, help="Target glass transition temperature")
    parser.add_argument("--target_Tm", type=float, help="Target melting temperature")
    parser.add_argument("--processing_temp", type=float, help="Processing temperature condition")
    parser.add_argument("--num_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    # Prepare target properties
    targets = {}
    if args.target_Tg:
        targets['Tg'] = args.target_Tg
    if args.target_Tm:
        targets['Tm'] = args.target_Tm
    
    # Prepare conditions
    conditions = None
    if args.processing_temp:
        conditions = {'temperature': args.processing_temp}
    
    # Generate
    polymers = inverse_design_polymers(
        args.model_path,
        targets,
        conditions,
        args.num_samples
    )
    
    print(f"Generated {len(polymers)} polymers:")
    for i, polymer in enumerate(polymers[:10]):
        print(f"{i+1}: {polymer}")
