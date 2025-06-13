import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import torch

# Patch Extraction

def extract_patches(image, patch_size=8, stride=8):
    """
    Görüntüden sabit boyutlu yama (patch) parçaları çıkarır.

    Args:
        image (torch.Tensor): 3 boyutlu bir görüntü tensörü [C, H, W]
        patch_size (int): Her bir yamanın kenar uzunluğu (varsayılan: 8)
        stride (int): Kaydırma miktarı (varsayılan: 8)

    Returns:
        List[torch.Tensor]: Görüntüden çıkarılan yamaların listesi
    """
    patches = []
    _, H, W = image.shape
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]  # [C, patch_size, patch_size]
            patches.append(patch)
    return patches

# Patch To Bytes

def patch_to_bytes(patch):
    """
    Patch tensörünü AES şifreleme için uygun byte formatına dönüştürür.

    Args:
        patch (torch.Tensor): [C, H, W] boyutlu tensör

    Returns:
        bytes: Tensörün byte biçimindeki karşılığı
    """
    patch_np = patch.cpu().numpy()     # Tensörü NumPy array'e çevir
    patch_bytes = patch_np.tobytes()   # NumPy array'i byte dizisine çevir
    return patch_bytes

# Key Generation

def generate_key(bits=128):
    """
    AES şifreleme için rastgele bir anahtar üretir.

    Args:
        bits (int): Anahtar uzunluğu (128 veya 256)

    Returns:
        bytes: AES anahtarı
    """
    assert bits in [128, 256], "AES supports 128 or 256 bits only!"
    return os.urandom(bits // 8)  # İstenen uzunlukta rastgele byte üret

# AES Encryption

def aes_encrypt(data_bytes, key):
    """
    Veriyi AES-ECB modunda şifreler.

    Args:
        data_bytes (bytes): Şifrelenecek veri (patch)
        key (bytes): AES anahtarı

    Returns:
        bytes: Şifrelenmiş veri (ciphertext)
    """
    cipher = AES.new(key, AES.MODE_ECB)              # ECB modu kullanılıyor
    ciphertext = cipher.encrypt(pad(data_bytes, AES.block_size))  # Blok boyutuna uygun hale getir ve şifrele
    return ciphertext
