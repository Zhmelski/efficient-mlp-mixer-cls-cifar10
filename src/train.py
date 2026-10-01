import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import os
from tqdm import tqdm

from dataset import get_dataloaders
from model import MLPMixer


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Используем tqdm для отображения прогресса по батчам
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Обновляем информацию в прогресс-баре
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # Гиперпараметры
    epochs = 50
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-2

    # Автоматический выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Создаем директорию для сохранения весов модели
    os.makedirs("./models", exist_ok=True)

    # Инициализация данных и модели
    train_loader, test_loader = get_dataloaders(batch_size=batch_size, data_dir='./data')
    model = MLPMixer().to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Планировщик скорости обучения (постепенно снижает lr по косинусоиде)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Настройка MLflow (используем локальную SQLite БД, чтобы не плодить тысячи мелких файлов в кэше)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MLP-Mixer_CIFAR10")

    with mlflow.start_run():
        # Логируем конфигурацию запуска
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "parameters_count": sum(p.numel() for p in model.parameters() if p.requires_grad)
        })

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            print(f"\nЭпоха {epoch}/{epochs}")

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

            # Логируем метрики в MLflow для отрисовки графиков
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": scheduler.get_last_lr()[0]
            }, step=epoch)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

            # Сохраняем веса, если точность на тесте улучшилась
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), "./models/best_mlp_mixer.pth")
                print(">>> Новая лучшая модель сохранена!")

            scheduler.step()


if __name__ == "__main__":
    main()