import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
import torch

# Patch çıkarma
def extract_patches(image, patch_size=8, stride=8):
    patches = []
    _, H, W = image.shape
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

# Patch to bytes
def patch_to_bytes(patch):
    patch_np = patch.cpu().numpy()
    patch_bytes = patch_np.tobytes()
    return patch_bytes

# AES anahtar üretimi
def generate_key(bits=128):
    assert bits in [128, 256], "AES supports 128 or 256 bits!"
    return os.urandom(bits // 8)

# AES şifreleme
def aes_encrypt(data_bytes, key):
    cipher = AES.new(key, AES.MODE_ECB)
    ciphertext = cipher.encrypt(pad(data_bytes, AES.block_size))
    return ciphertext
