#!/usr/bin/env python3

import torch
import argparse
import yaml

def load_config_and_model(config_filepath):
    # Load the configuration file
    with open(config_filepath, 'r') as file:
        config = yaml.safe_load(file)

    # Extract model information
    metric_3d_model = config['metric_3d_model']

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('yvanyin/metric3d', metric_3d_model, pretrain=True, map_location=device)
    model.eval()
    model = model.to(device)
    print(f"Model {metric_3d_model} loaded successfully on {device}.")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download depth estimation model.')
    parser.add_argument('--config', type=str, help='Path to configuration file.')
    args = parser.parse_args()
    config_filepath = args.config
    model = load_config_and_model(config_filepath)