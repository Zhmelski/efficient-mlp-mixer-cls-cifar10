import torch
import torch.nn as nn


class MLP(nn.Module):
    """Стандартный полносвязный блок из двух линейных слоев."""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(nn.Module):
    """Один блок архитектуры MLP-Mixer."""

    def __init__(self, num_patches, hidden_dim, tokens_hidden_dim, channels_hidden_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        # Token-mixing работает с патчами (перемешивает пространственную информацию)
        self.token_mix = MLP(num_patches, tokens_hidden_dim, num_patches, dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        # Channel-mixing работает внутри каждого патча (извлекает признаки)
        self.channel_mix = MLP(hidden_dim, channels_hidden_dim, hidden_dim, dropout)

    def forward(self, x):
        # x имеет форму: [Batch, Patches, HiddenDim]

        # 1. Token mixing
        y = self.norm1(x)
        y = y.transpose(1, 2)  # Транспонируем для линейного слоя: [Batch, HiddenDim, Patches]
        y = self.token_mix(y)
        y = y.transpose(1, 2)  # Возвращаем обратно: [Batch, Patches, HiddenDim]
        x = x + y  # Residual connection

        # 2. Channel mixing
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y  # Residual connection

        return x


class MLPMixer(nn.Module):
    def __init__(
            self,
            image_size=32,
            channels=3,
            patch_size=4,
            num_classes=10,
            hidden_dim=128,
            num_blocks=6,
            tokens_hidden_dim=64,
            channels_hidden_dim=512,
            dropout=0.1
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Размер изображения должен быть кратен размеру патча."

        self.num_patches = (image_size // patch_size) ** 2  # Для 32x32 и патча 4x4 -> 64 патча
        self.patch_dim = channels * patch_size ** 2  # Для RGB и 4x4 -> 3 * 16 = 48
        self.patch_size = patch_size

        # Строго используем nn.Linear для проекции патчей (вместо Conv2d)
        self.patch_embedding = nn.Linear(self.patch_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            MixerBlock(
                num_patches=self.num_patches,
                hidden_dim=hidden_dim,
                tokens_hidden_dim=tokens_hidden_dim,
                channels_hidden_dim=channels_hidden_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Вход: [B, 3, 32, 32]
        B, C, H, W = x.shape
        P = self.patch_size

        # Разрезаем изображение на патчи исключительно тензорными операциями
        # [B, 3, 32, 32] -> [B, 64, 48]
        x = x.view(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, -1, self.patch_dim)

        # Проецируем патчи (Linear)
        x = self.patch_embedding(x)  # -> [Batch, 64, 128]

        # Прогоняем через блоки Mixer
        for block in self.blocks:
            x = block(x)

        # Усредняем признаки по всем патчам (Global Average Pooling)
        x = self.norm(x)
        x = x.mean(dim=1)  # -> [Batch, 128]

        # Финальная классификация (Linear)
        x = self.head(x)  # -> [Batch, 10]

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLPMixer()
    print(f"Количество обучаемых параметров: {count_parameters(model):,}")

    # Имитация одного батча изображений
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    print(f"Форма выхода: {output.shape} (Ожидается [2, 10])")