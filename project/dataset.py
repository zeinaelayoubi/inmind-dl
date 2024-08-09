import os
from torch.utils.data import Dataset
import numpy as np 
from PIL import Image
class CustomImageDataset(Dataset):
    def __init__(self, mask_dir, images_dir, transform=None, target_transform=None):
        self.masks = mask_dir
        self.images = images_dir
        self.transform = transform
        self.files=os.listdir(images_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        img_path=os.path.join(self.images,self.files[index])
        mask_path=os.path.join(self.masks,self.files[index].replace('.png', '_mask.gif'))
        image=np.array(Image.open(img_path).convert("RGB"))
        mask=p.array(Image.open(mask_path).convert("L"))
        mask[mask==255]=1
        
            
        #self.tranform is not None:
        #do transformation
        return image,mask      
      

#load the paths 
