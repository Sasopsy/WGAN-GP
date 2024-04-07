from PIL import Image
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Resize, ToTensor, Compose
import train


transforms = Compose([
    ToTensor(),
    Resize(train.TrainConfig().image_shape),  # It has to torch.Tensor or PIL image
])


class Landscape(Dataset):
    
    def __init__(self,
                 train_config: train.TrainConfig,
                 transform = transforms):
        super(Landscape,self).__init__()
        self.root_directory = train_config.root_directory
        self.transform = transform
        
        # Get its item list.
        self.dataset = os.listdir(self.root_directory)
        if '.DS_Store' in self.dataset:  # For MacOS.
            self.dataset.remove('.DS_Store') 
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, image_index: int) -> torch.Tensor:
        self.image_index = image_index
        image_path = os.path.join(self.root_directory,self.dataset[image_index])
        image = np.array(Image.open(image_path).convert("RGB"))
        
        if self.transform:
            image = self.transform(image)
            
        return image
        