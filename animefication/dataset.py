
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import config

class PhotoDataset(Dataset):
    def __init__(self, root_dir=config.ROOT_DIR, root_A="trainA", root_B="trainB", transform=None):
        self.root_dir = root_dir
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform
        self.image_files_A = [f for f in os.listdir(os.path.join(root_dir, self.root_A)) if f.endswith('.jpg')]
        self.image_files_B = [f for f in os.listdir(os.path.join(root_dir, self.root_B)) if f.endswith('.jpg')]
        self.monet_length = len(self.image_files_A)
        self.photos_length = len(self.image_files_B)

    def __len__(self):
        return max(self.monet_length, self.photos_length)

    def __getitem__(self, idx):
        img_name_A = os.path.join(self.root_dir, self.root_A, self.image_files_A[idx % len(self.image_files_A)])
        img_name_B = os.path.join(self.root_dir, self.root_B, self.image_files_B[idx % len(self.image_files_B)])
        # idx % len to avoid situation when our idx greater than lenght of dataset. Using this we can cycle our dataset whithin big number of epochs

        
        image_A = Image.open(img_name_A).convert('RGB')
        image_B = Image.open(img_name_B).convert('RGB')

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(*config.STATS),
])

def denorm(img_tensors):
    return img_tensors * config.STATS[1][0] + config.STATS[0][0]