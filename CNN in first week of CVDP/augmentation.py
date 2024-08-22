import cv2
import os
import albumentations as A

def draw_bounding_boxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image.
    
    Args:
        image (ndarray): The image on which to draw.
        bboxes (list): List of bounding boxes in YOLO format (x_center, y_center, width, height).
        color (tuple): Color of the bounding box lines.
        thickness (int): Thickness of the bounding box lines.
    
    Returns:
        ndarray: Image with bounding boxes drawn.
    """
    height, width = image.shape[:2]
    for bbox in bboxes:
        x_center, y_center, bbox_width, bbox_height = bbox
        # Convert from YOLO format to absolute coordinates
        x1 = int((x_center - bbox_width / 2) * width)
        y1 = int((y_center - bbox_height / 2) * height)
        x2 = int((x_center + bbox_width / 2) * width)
        y2 = int((y_center + bbox_height / 2) * height)
        
        # Draw the bounding box
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

images_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\data\images'
labels_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\data\labels'
augmented_images_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\augmented\images'
augmented_labels_path = r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\augmented\labels'

# Define your augmentation pipeline here
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

for filename in os.listdir(images_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        image_file = os.path.join(images_path, filename)
        label_file = os.path.join(labels_path, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Read image
        image = cv2.imread(image_file)
        height, width = image.shape[:2]
        
        # Read YOLO format label
        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        # Parse the YOLO format labels
        bboxes = []
        class_labels = []
        for line in label_lines:
            if line.strip():  # Check if the line is not empty
                parts = line.split()
                if len(parts) == 5:  # Ensure there are 5 values to unpack
                    try:
                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        bboxes.append([x_center, y_center, bbox_width, bbox_height])
                        class_labels.append(int(class_id))
                    except ValueError:
                        print(f"Skipping invalid line in {label_file}: {line.strip()}")
                        continue

        if not bboxes:
            continue  # Skip if there are no valid bounding boxes
        
        # Draw bounding boxes on original image
        image_with_boxes = draw_bounding_boxes(image.copy(), bboxes)
        
        # Apply augmentation
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        
        # Draw bounding boxes on augmented image
        transformed_image_with_boxes = draw_bounding_boxes(transformed_image.copy(), transformed_bboxes)
        
        # Save the augmented images with bounding boxes
        augmented_image_file = os.path.join(augmented_images_path, filename)
        cv2.imwrite(augmented_image_file, transformed_image_with_boxes)
        
        # Save the augmented label
        augmented_label_file = os.path.join(augmented_labels_path, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(augmented_label_file, 'w') as f:
            for bbox, class_id in zip(transformed_bboxes, class_labels):
                x_center, y_center, bbox_width, bbox_height = bbox
                # Write to file in YOLO format
                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
