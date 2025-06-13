# src/train_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.cnn_model import CNN

if __name__ == "__main__":
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # Veri seti (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # Model
    cnn_model = CNN().to(device)

    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

    # Eğitim
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Model kaydet
    torch.save(cnn_model.state_dict(), 'models/cnn_model.pth')
    print("✅ Model kaydedildi: models/cnn_model.pth")
