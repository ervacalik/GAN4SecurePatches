import sys
import os

# Proje kök dizinini sys.path'e ekle → src klasörüne erişim sağlanır
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Gerekli kütüphaneler
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.cnn_model import CNN
from src.gan_model import Generator, Discriminator
from src.gradcam import GradCAM
from src.utils import extract_patches, patch_to_bytes, generate_key, aes_encrypt
import torch.nn as nn
import torch.optim as optim
import time

# GPU varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eğitilmiş CNN modelini yükle (sınıflandırma ve GradCAM için)
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=torch.device('cpu')))
cnn_model.eval()

# PSNR ve SSIM hesaplamak için yardımcı fonksiyon
def calculate_metrics(real_patch, gan_patch):
    real_patch_norm = (real_patch - real_patch.min()) / (real_patch.max() - real_patch.min())
    psnr = peak_signal_noise_ratio(real_patch_norm, gan_patch, data_range=1.0)
    ssim = structural_similarity(
        np.transpose(real_patch_norm, (1,2,0)),
        np.transpose(gan_patch, (1,2,0)),
        win_size=3,
        channel_axis=2,
        data_range=1.0
    )
    return psnr, ssim

# Streamlit UI
st.title("🔐 Görüntü Patch Şifreleme & GAN Kurtarma Demo")
st.markdown("""
Bu uygulama, **GradCAM tabanlı adaptif AES şifreleme** ve **GAN ile patch recovery** sürecini gösterir.
""")
# Görsel yükleme adımı
st.header("📥 1️⃣ Görsel Yükleme")
uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Yüklenen görüntüyü RGB formatına çevirir
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

    # Görüntüyü CNN boyutuna uyarlamak için dönüşüm uygula
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # GradCAM ile anlamlı bölgeleri çıkar
    st.header("🔍 2️⃣ GradCAM Çıktısı")
    label = 0  # Örnek sınıf index( manuel belirlenmiş)
    gradcam = GradCAM(cnn_model, cnn_model.conv2)
    heatmap = gradcam(image_tensor, target_class=label)

    # Görselleştir
    fig, ax = plt.subplots()
    ax.imshow(np.transpose((image_tensor[0].cpu().numpy() * 0.5 + 0.5), (1,2,0)))
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    st.pyplot(fig)

    # AES-128 / AES-256 ile patch bazlı şifreleme
    st.header("🔐 3️⃣ Adaptif Şifreleme Süresi")
    patches = extract_patches(image_tensor[0].cpu(), patch_size=8, stride=8)
    key_128 = generate_key(128)
    key_256 = generate_key(256)

    importance_threshold = 0.5
    patch_size = 8
    stride = 8
    H, W = heatmap.shape

    encrypted_patches = []
    start_time = time.time()

    idx = 0
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = patches[idx]
            idx += 1

            patch_heatmap = heatmap[y:y+patch_size, x:x+patch_size]
            patch_importance = patch_heatmap.mean()
            patch_bytes = patch_to_bytes(patch)

            # Önemli patch → AES-256, diğerleri → AES-128
            if patch_importance > importance_threshold:
                ciphertext = aes_encrypt(patch_bytes, key_256)
            else:
                ciphertext = aes_encrypt(patch_bytes, key_128)

            encrypted_patches.append(ciphertext)

    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    st.write(f"Adaptif Şifreleme Süresi: **{total_time:.2f} ms**")

    # GAN Eğitimi ve Sonuçların Karşılaştırılması
    # GAN Eğitme bölümü
    st.header("🤖 4️⃣ GAN Eğitimi ve Karşılaştırma")

    if st.button("GAN Eğit ve Tahmin Yap"):
        # GAN modelini oluştur
        G = Generator().to(device)
        D = Discriminator().to(device)

        loss_fn = nn.BCELoss()
        g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
        d_optimizer = optim.Adam(D.parameters(), lr=0.0002)

        # Patchleri flatten et
        real_patch_tensors = []
        for patch in patches:
            real_patch = patch.reshape(-1)
            real_patch_tensors.append(real_patch)

        real_patch_dataset = torch.stack(real_patch_tensors).to(device)
        num_patches = len(patches)

        # Eğitim parametreleri
        epochs = 30
        batch_size = 16

        progress_bar = st.progress(0)
        status_text = st.empty()

        for epoch in range(epochs):
            permutation = torch.randperm(num_patches)
            for i in range(0, num_patches, batch_size):
                indices = permutation[i:i+batch_size]
                real_patches_batch = real_patch_dataset[indices]
                batch_size_curr = real_patches_batch.size(0)

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

            progress_bar.progress((epoch+1) / epochs)
            status_text.text(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

        # Eğitim tamamlandığında ilk patch'in tahminini yap
        real_patch = patches[0].cpu().numpy()
        noise = torch.randn((1, 8*8*3), device=device)
        with torch.no_grad():
            output = G(noise).view(3,8,8).cpu().numpy()
            output = (output - output.min()) / (output.max() - output.min())

        # PSNR & SSIM hesapla
        psnr, ssim = calculate_metrics(real_patch, output)

        # Görselleştirme
        fig, axs = plt.subplots(1, 2, figsize=(6,3))
        axs[0].imshow(np.transpose(real_patch, (1,2,0)))
        axs[0].set_title("Orijinal Patch")
        axs[0].axis("off")

        axs[1].imshow(np.transpose(output, (1,2,0)))
        axs[1].set_title(f"GAN Tahmini\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
        axs[1].axis("off")

        st.pyplot(fig)

