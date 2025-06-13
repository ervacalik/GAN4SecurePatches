# train_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

from src.gan_model import Generator, Discriminator
from src.utils import extract_patches

from src.cnn_model import CNN  # sadece patch extraction için
import torchvision.datasets as datasets

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Veri seti yükle
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Patchleri hazırlayacağız
# Basitçe bir batch'ten ilk image'i alıp patch extraction yapacağız
dataiter = iter(train_loader)
images, _ = next(dataiter)
image = images[0]

patches = extract_patches(image, patch_size=8, stride=8)
num_patches = len(patches)

# Patchleri flatten edelim
def flatten_patch(patch):
    return patch.reshape(-1)

real_patch_tensors = []
for patch in patches:
    real_patch = flatten_patch(patch)
    real_patch_tensors.append(real_patch)

real_patch_dataset = torch.stack(real_patch_tensors).to(device)

# GAN modellerini oluştur
G = Generator().to(device)
D = Discriminator().to(device)

# Loss ve optimizer
loss_fn = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)

# Eğitim parametreleri
epochs = 50
batch_size = 16

# Eğitim döngüsü
for epoch in range(epochs):
    permutation = torch.randperm(num_patches)
    for i in range(0, num_patches, batch_size):
        indices = permutation[i:i+batch_size]
        real_patches_batch = real_patch_dataset[indices]
        batch_size_curr = real_patches_batch.size(0)

        # Rastgele noise
        noise = torch.randn((batch_size_curr, 8*8*3), device=device)

        # Discriminator eğitimi
        D.zero_grad()
        real_label = torch.ones((batch_size_curr, 1), device=device)
        fake_label = torch.zeros((batch_size_curr, 1), device=device)

        output_real = D(real_patches_batch)
        d_loss_real = loss_fn(output_real, real_label)

        generated_patch = G(noise).detach()
        output_fake = D(generated_patch)
        d_loss_fake = loss_fn(output_fake, fake_label)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Generator eğitimi
        G.zero_grad()
        generated_patch = G(noise)
        output = D(generated_patch)
        g_loss = loss_fn(output, real_label)
        g_loss.backward()
        g_optimizer.step()

    # Epoch sonucu yazdır
    print(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

# Generator modelini kaydet
torch.save(G.state_dict(), 'models/gan_generator.pth')
print("✅ GAN Generator modeli kaydedildi: models/gan_generator.pth")
