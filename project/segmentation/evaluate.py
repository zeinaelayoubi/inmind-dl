import json
import numpy as np

# Path to your JSON file
json_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\json\semantic_segmentation_labels_0000.json'

# Load the JSON labels
with open(json_path, 'r') as f:
    labels = json.load(f)
    
print("labels", labels)