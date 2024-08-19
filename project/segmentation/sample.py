import torch
from torchvision import transforms
from dataset import SegmentationDataset  # Assuming this is your dataset class
from utils import visualize_image_and_mask

# Define your transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
])

# Create dataset
dataset = SegmentationDataset(
        image_dir=r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\images',
        mask_dir=r'C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\dataset\semantic_segmentation\train\masks',
    transform=transform
)

# Define the index of the sample to visualize
sample_index = 24  # Change this index to view different samples

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Visualize the image and mask at the specified index
visualize_image_and_mask(dataset, sample_index, title='Sample Visualization')
