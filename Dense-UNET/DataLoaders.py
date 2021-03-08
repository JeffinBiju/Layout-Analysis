from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import glob
import torchvision.transforms as T
import os

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.8061871094330194, 0.8237343918188842, 0.8058428615017786),
                (0.2430262593379671, 0.24955391590832207, 0.2624875767860221))
])


class PrimaLayout(Dataset):
    def __init__(self, root):
        self.paths = glob.glob(root + '/NPmasks/*.npy')
        self.img_paths = []
        self.npy_paths = []
        for item in self.paths:
            img_path = item.replace('NPmasks', 'Images2').replace('.npy', '.jpg')
            if os.path.exists(img_path):
                self.img_paths.append(img_path)
                self.npy_paths.append(item)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        target = np.load(self.npy_paths[idx])
        target = torch.FloatTensor(target)

        image = Image.open(self.img_paths[idx])
        image = transform(image)
        filename = self.img_paths[idx].split('/')[-1]
        return image, target, filename


dataset = PrimaLayout('.')

batch_size = 6
validation_split = .1
shuffle_dataset = True
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=2)
validation_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=4, num_workers=2)

print(dataset_size)
