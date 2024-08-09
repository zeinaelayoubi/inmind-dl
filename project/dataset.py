import os
from torch.utils.data import Dataset
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, mask_dir, images_dir, transform=None):
        self.mask_dir = mask_dir
        self.images_dir = images_dir
        self.transform = transform
        self.files = os.listdir(images_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.files[index])
        mask_path = os.path.join(self.mask_dir, self.files[index].replace('.png', '_mask.gif'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert mask to binary (assuming 255 indicates the object)
        mask = np.array(mask)
        mask[mask == 255] = 1

        
        if self.transform:
            image = self.transform(image)

        # Convert mask back to PIL Image for further processing, if necessary
        mask = Image.fromarray(mask)

        return image, mask


mask_dir = "C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train_masks"
images_dir = "C:\Users\Personal\OneDrive - Lebanese American University\inmind\Inmind_workspace\project\train"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


dataset = CustomImageDataset(mask_dir=mask_dir, images_dir=images_dir, transform=transform)
