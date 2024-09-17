# This script is used to save the LART bounding box visuals
# One image per bounding box
# Filename will be {sample_id}_{tracking_id}.png

# Lart is a {sample_id}.pkl file with the following structure:
# tracking_id -> {bbox2d}

# Args will be the path to the LART data and the path to the corresponding images
# Output will be saved to {output_dir}/lart_bbox_visuals

import os
import json
import numpy as np
import cv2
from PIL import Image
import pickle
import tqdm

def save_all_lart_bounding_boxes_for_sample(image_filepath: str, lart_data_path: str, output_dir: str):
    # Load the LART data
    with open(lart_data_path, 'rb') as f:
        lart_data = pickle.load(f)
        
    sample_id = os.path.basename(lart_data_path).split('.')[0]
        
    for tracking_id in lart_data:
        bbox = lart_data[tracking_id]['bbox']
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Load the image
        assert os.path.exists(image_filepath), f"Image file {image_filepath} does not exist"
        image = cv2.imread(image_filepath)
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Save the image
        output_filepath = os.path.join(output_dir, f"{sample_id}_{tracking_id}.png")
        cv2.imwrite(output_filepath, image)
        
def save_all_visuals(lart_data_dir: str, images_dir: str, output_dir: str):
    # only save first 5
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def is_desired_lart_file(filename: str) -> bool:
        if filename.endswith('_907.pkl'):
            return True
        elif filename.endswith('_997.pkl'):
            return True
        elif filename.endswith('_1054.pkl'):
            return True
        elif filename.endswith('_1157.pkl'):
            return True
        elif filename.endswith('_1174.pkl'):
            return True
        return False

    lart_data_paths = os.listdir(lart_data_dir)
    lart_data_paths = [lart_data_path for lart_data_path in lart_data_paths if is_desired_lart_file(lart_data_path)]

    for lart_data_path in tqdm.tqdm(lart_data_paths):
        image_filename = lart_data_path.split('.')[0] + '.png'
        image_filepath = os.path.join(images_dir, image_filename)
        lart_data_path = os.path.join(lart_data_dir, lart_data_path)
        save_all_lart_bounding_boxes_for_sample(image_filepath, lart_data_path, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save LART bounding box visuals")
    parser.add_argument("--lart_data_dir", type=str, help="Path to the LART data directory")
    parser.add_argument("--images_dir", type=str, help="Path to the images directory")
    parser.add_argument("--output_dir", type=str, help="Path to save the output")
    args = parser.parse_args()
    save_all_visuals(args.lart_data_dir, args.images_dir, args.output_dir)
    

    
