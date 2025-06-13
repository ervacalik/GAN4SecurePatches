import torch.nn as nn

class CNN(nn.Module):
    
    """
        Bu sınıf, görüntü sınıflandırma için kullanılan bir Konvolüsyonel Sinir Ağı (CNN) mimarisini tanımlar.
        Bu model, CIFAR-10 gibi 32x32 boyutlu, 3 kanallı (RGB) görüntüler üzerinde çalışmak üzere optimize edilmiştir.
        
        Bu CNN modeli GradCAM algoritması ile birlikte çalışarak görüntünün önemli bölgelerini belirlemek için de kullanılmaktadır.
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # 1. Konvolüsyon Katmanı:
        # Giriş: 3 kanal (RGB) / Çıkış: 32 kanal / Filtre: 3x3 / Padding: kenar bilgisi kaybolmasın diye 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # Aktivasyon fonksiyonu
        self.relu = nn.ReLU()
        
        # MaxPooling: 2x2'lik alandan maksimum değeri alır, boyutu yarıya düşürür (32x32 → 16x16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Konvolüsyon Katmanı:
        # Giriş: 32 kanal / Çıkış: 64 kanal / Filtre: 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Dropout katmanı: Overfitting’i önlemek için %20 nöronu eğitim sırasında rastgele devre dışı bırakır
        self.dropout = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        """
            Modelin ileri (forward) geçişini tanımlar.
            Girişten başlayarak her bir katmandan verinin nasıl geçtiğini belirtir.
        """
        
        # 1. konvolüsyon + ReLU + MaxPool → Boyut: 32x32 → 16x16
        x = self.pool(self.relu(self.conv1(x)))
        
        # 2. konvolüsyon + ReLU + MaxPool → Boyut: 16x16 → 8x8
        x = self.pool(self.relu(self.conv2(x)))
        
        # Tensor'u tam bağlantılı katmanlara verebilmek için düzleştiriyoruz: [batch_size, 4096]
        x = x.view(-1, 64*8*8)
        
        # Dropout + ReLU + FC katmanı → boyut: 4096 → 128
        x = self.dropout(self.relu(self.fc1(x)))
        
        # Son katman: 128 → 10 (her sınıf için bir skor)
        x = self.fc2(x)
        return x
