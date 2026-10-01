import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


class Cifar10Albumentations(Dataset):
    def __init__(self, root_dir, train=True, download=True):
        self.cifar10 = datasets.CIFAR10(root=root_dir, train=train, download=download)
        self.train = train

        # Задаем средние значения и стандартное отклонение для CIFAR-10 (нормализация)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        # Вариант предварительной обработки (Кадрирование + Отражение)
        if self.train:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=36, min_width=36, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomCrop(height=32, width=32),
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range=(8, 8),
                    hole_width_range=(8, 8),
                    fill=mean,
                    p=0.5
                ),  # Cutout для лучшей регуляризации
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            # Для тестовой выборки только нормализация
            self.transform = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        img, label = self.cifar10[idx]
        img = np.array(img)  # Переводим PIL Image в NumPy array для Albumentations

        augmented = self.transform(image=img)
        img_tensor = augmented['image']

        return img_tensor, label


def get_dataloaders(batch_size=128, data_dir='./data'):
    train_dataset = Cifar10Albumentations(root_dir=data_dir, train=True, download=True)
    test_dataset = Cifar10Albumentations(root_dir=data_dir, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# Небольшой блок для проверки работы
if __name__ == "__main__":
    train_dl, test_dl = get_dataloaders(batch_size=32)
    images, labels = next(iter(train_dl))
    print(f"Shape батча изображений: {images.shape}")  # Ожидаем [32, 3, 32, 32]
    print(f"Shape батча меток: {labels.shape}")