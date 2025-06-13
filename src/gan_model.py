import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        """
            GENERATOR: Gürültüden (noise) gerçek gibi görünen patch'ler üretir.
        """
        self.model = nn.Sequential(
            nn.Linear(8*8*3, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 8*8*3),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """
            DISCRIMINATOR: Girdi patch'inin gerçek mi sahte mi olduğunu ayırt etmeye çalışır.
        """
        self.model = nn.Sequential(
            nn.Linear(8*8*3, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

"""
    Bu sınıflar sayesinde proje, sadece şifreleme değil, 
    şifre çözümü olmayan durumlar için de tahmine dayalı kurtarma (recovery) yapabilmektedir.
"""