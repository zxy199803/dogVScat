from config import Config
import torchvision
import os
import torch
from PIL import Image

my_config = Config()  # 实例化配置文件


class Dataset:
    def __init__(self, train=True):
        # 图像预处理
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((my_config.input_size, my_config.input_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(my_config.mean, my_config.std)
            ]
        )
        self.train = train

        if train:
            self.data_images = {
                'train': torchvision.datasets.ImageFolder(root=os.path.join(my_config.data_root, 'train'),
                                                          transform=self.transform),
                'valid': torchvision.datasets.ImageFolder(root=os.path.join(my_config.data_root, 'valid'),
                                                          transform=self.transform),
            }
            self.data_images_loader = {
                'train': torch.utils.data.DataLoader(dataset=self.data_images['train'], batch_size=my_config.batch_size,
                                                     shuffle=True),
                'valid': torch.utils.data.DataLoader(dataset=self.data_images['valid'], batch_size=my_config.batch_size,
                                                     shuffle=True)
            }

            self.classes = self.data_images['train'].classes  # ['cat','dog']
            self.classes_index = self.data_images['train'].class_to_idx  # {'cat': 0, 'dog': 1}
        else:
            images = [os.path.join(my_config.data_test_root, img) for img in os.listdir(my_config.data_test_root)]
            self.images = sorted(images, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

    def __getitem__(self, index):
        img_path = self.images[index]
        label = int(self.images[index].split('.')[-2].split('/')[-1])
        data_images_test = Image.open(img_path)
        data_images_test = self.transform(data_images_test)
        return data_images_test, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = Dataset()
