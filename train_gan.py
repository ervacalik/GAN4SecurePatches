
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

from src.gan_model import Generator, Discriminator
from src.utils import extract_patches
from src.cnn_model import CNN  # Sadece örnek görsel almak için
import torchvision.datasets as datasets

# Donanım tercihi (GPU varsa kullanılır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# CIFAR-10 veri seti hazırlanır
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Görüntüleri -1 ila 1 aralığına çeker
])

# Eğitim veri seti yüklenir
train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Sadece bir görsel alınarak patch'ler çıkarılır
dataiter = iter(train_loader)
images, _ = next(dataiter)
image = images[0]

# 8x8 boyutunda, stride=8 olacak şekilde yama çıkarımı yapılır
patches = extract_patches(image, patch_size=8, stride=8)
num_patches = len(patches)

# Patch'leri vektöre dönüştür
def flatten_patch(patch):
    return patch.reshape(-1)

real_patch_tensors = []
for patch in patches:
    real_patch = flatten_patch(patch)
    real_patch_tensors.append(real_patch)

# Gerçek patch veriseti oluşturulur
real_patch_dataset = torch.stack(real_patch_tensors).to(device)

# Generator ve Discriminator oluşturulur
G = Generator().to(device)
D = Discriminator().to(device)

# Kayıp fonksiyonu ve optimizasyon algoritmaları
loss_fn = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)

# Eğitim parametreleri
epochs = 50
batch_size = 16

# GAN eğitim döngüsü
for epoch in range(epochs):
    # Veriler rastgele karıştırılır
    permutation = torch.randperm(num_patches)

    for i in range(0, num_patches, batch_size):
        indices = permutation[i:i+batch_size]
        real_patches_batch = real_patch_dataset[indices]
        batch_size_curr = real_patches_batch.size(0)

        # Rastgele gürültü (Generator girişi)
        noise = torch.randn((batch_size_curr, 8*8*3), device=device)

        # Discriminator eğitimi
        D.zero_grad()
        real_label = torch.ones((batch_size_curr, 1), device=device)
        fake_label = torch.zeros((batch_size_curr, 1), device=device)

        # Gerçek patch'lerle discriminator kaybı
        output_real = D(real_patches_batch)
        d_loss_real = loss_fn(output_real, real_label)

        # Sahte patch'lerle discriminator kaybı
        generated_patch = G(noise).detach()
        output_fake = D(generated_patch)
        d_loss_fake = loss_fn(output_fake, fake_label)

        # Toplam discriminator kaybı ve optimizasyon
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Generator eğitimi
        G.zero_grad()
        generated_patch = G(noise)
        output = D(generated_patch)
        g_loss = loss_fn(output, real_label)  # Amaç: discriminator'u kandırmak
        g_loss.backward()
        g_optimizer.step()

    # Her epoch sonunda kayıplar yazdırılır
    print(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

# Eğitilmiş Generator modeli diske kaydedilir
torch.save(G.state_dict(), 'models/gan_generator.pth')
print("GAN Generator modeli kaydedildi: models/gan_generator.pth")
