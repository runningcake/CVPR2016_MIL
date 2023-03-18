import torch
from torch.utils import data
import torchvision.transforms as T
from PIL import Image


class DIYError(Exception):
    pass

class MyDataset(data.Dataset):
    """
    Args:
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self,
                 transform=T.ToTensor(),
                 paths=None):
        self.transform = transform
        if not paths: # []
            raise DIYError("paths is empty.")
        self.images = paths
        self.labels = torch.tensor([int(path.split("/")[-1][5]) for path in paths])
        class_cnt_inv = torch.tensor([1 / (self.labels == i).sum() for i in range(3)])
        self.weights = class_cnt_inv / class_cnt_inv.sum()
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, path).
        """
        path = self.images[index]
        img = Image.open(path).convert('RGB')
        label = self.labels[index]
        img = self.transform(img)
        return img, label, path

    def __len__(self):
        return len(self.images)