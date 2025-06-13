# GAN4SecurePatches

# 🔐 GradCAM Destekli Görüntü Şifreleme ve GAN ile Onarma Sistemi

Bu proje, görüntülerdeki anlamlı bölgelerin **Grad-CAM** ile belirlenip adaptif olarak **AES-128/256** algoritmalarıyla şifrelenmesini ve bu şifreli yama (patch) verilerin **GAN** yardımıyla yeniden oluşturulmasını hedefler. Ayrıca **CNN** modeli ile sınıflandırma yapılır ve **Grad-CAM** ile modelin dikkat ettiği alanlar görselleştirilir.


![Uygulama](example_gradcam.gif)

## 🏗️ Teknik Mimari

### 🔹 Genel Akış
Görüntü → CNN → Grad-CAM → Patch Extraction → AES Şifreleme → GAN ile Onarma → PSNR/SSIM Hesaplama

### 🔹 CNN + GradCAM
- Basit bir CNN ile sınıflandırma
- `conv2` katmanından Grad-CAM aktivasyonları çıkarılır
- Sınıfa özel dikkat haritası üretilir

### 🔹 AES Şifreleme (Adaptif)
- Grad-CAM skoruna göre patch'ler önemli / önemsiz olarak etiketlenir
- Önemli patch'ler: **AES-256**
- Önemsiz patch'ler: **AES-128**
- Klasik tek tip şifreleme ile karşılaştırma yapılır

### 🔹 GAN ile Patch Kurtarma
- Generator: Patch benzeri örnek üretir (8×8 RGB)
- Discriminator: Gerçek vs sahte ayrımı yapar
- Girdi: Noise vektör (şifreli veri yerine)
- Eğitim sonrası GAN, patch'leri geri üretir

### 🔹 Kalite Ölçüm
- `PSNR (Peak Signal to Noise Ratio)`
- `SSIM (Structural Similarity Index)`

## 📁 Dosya Yapısı
GAN4SecurePatches/
├── models/
│ ├── cnn_model.pth
│ └── gan_generator.pth
├── src/
│ ├── cnn_model.py
│ ├── gan_model.py
│ ├── gradcam.py
│ └── utils.py
├── app.py # Streamlit demo arayüzü
├── train_cnn.py # CNN eğitim scripti
├── train_gan.py # GAN eğitim scripti
├── requirements.txt
├── README.md
└── docs/
├── example_gradcam.gif

---

## 💻 Kurulum ve Gereksinimler

### 📦 Bağımlılıklar
- Python 3.9+
- PyTorch
- torchvision
- Streamlit
- matplotlib
- numpy
- pycryptodome
- scikit-image

### 🧰 Sanal Ortam (Önerilir)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows


pip install -r requirements.txt










