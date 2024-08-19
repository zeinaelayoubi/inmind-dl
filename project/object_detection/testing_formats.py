import json
import numpy as np

# Path to your JSON file
json_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\train\bounding_box_2d_tight_labels_0000.json'

# Load the JSON labels
with open(json_path, 'r') as f:
    labels = json.load(f)
    
print("labels", labels)

print("---------------")

# Path to your bounding boxes .npy file
bbox_file = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\object_detection\train\bounding_box_2d_tight_0000.npy'

# Load the bounding boxes
bboxes = np.load(bbox_file, allow_pickle=True)

print("bbox", bboxes)

# No need to reshape as each element is already a tuple with 6 values
print(f'Number of bounding boxes: {len(bboxes)}')

print("---------------")


print(bboxes.shape)
print(type(bboxes))

# Iterate through the bounding boxes and print each one with its corresponding label
for bbox in bboxes:
    class_id = int(bbox[0])  # Get the class ID
    class_label = labels[str(class_id)]['class']  # Get the class label from the JSON
    print(f'Class: {class_label}')
