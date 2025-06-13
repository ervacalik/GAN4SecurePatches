
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.cnn_model import CNN  # Kendi tanımladığın CNN mimarisi

if __name__ == "__main__":
    # Donanım seçimi (eğer varsa GPU kullanılır)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # CIFAR-10 veri seti için dönüşüm tanımlanır
    transform = transforms.Compose([
        transforms.ToTensor(),  # Görüntüyü tensöre çevirir
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizasyon işlemi
    ])

    # Eğitim veri seti CIFAR-10'dan yüklenir
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Veri seti DataLoader ile batch'ler halinde GPU/CPU'ya yüklenir
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True
    )

    # CNN modeli başlatılır
    cnn_model = CNN().to(device)

    # Kayıp fonksiyonu ve optimizer tanımlanır
    criterion = nn.CrossEntropyLoss()  # Sınıflandırma için uygun kayıp fonksiyonu
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

    # Eğitim döngüsü
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            # Girişler ve etiketler cihaza aktarılır
            inputs, labels = inputs.to(device), labels.to(device)

            # Gradyanlar sıfırlanır
            optimizer.zero_grad()

            # Modelden çıktı alınır
            outputs = cnn_model(inputs)

            # Kayıp hesaplanır
            loss = criterion(outputs, labels)

            # Geri yayılım yapılır
            loss.backward()

            # Ağırlıklar güncellenir
            optimizer.step()

            # Kayıp değeri toplanır
            running_loss += loss.item()

        # Epoch başına ortalama kayıp yazdırılır
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Eğitim tamamlandığında model diske kaydedilir
    torch.save(cnn_model.state_dict(), 'models/cnn_model.pth')
    print("Model kaydedildi: models/cnn_model.pth")
