import json

# Path to one of your JSON files
json_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\labels\semantic_segmentation_labels_0001.json'

with open(json_path, 'r') as f:
    label_data = json.load(f)

print(label_data)