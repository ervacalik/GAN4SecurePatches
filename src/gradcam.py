import torch
import torch.nn.functional as F

class GradCAM:

    """
    GradCAM (Gradient-weighted Class Activation Mapping) sınıfı,
    bir CNN modelinin hangi bölgelere dikkat ettiğini (feature importance)
    görselleştirmek için kullanılır.

    Bu sınıf, hedeflenen katmandaki aktivasyonları ve gradyanları kaydeder;
    ardından bu bilgilerle sınıf bazlı bir ısı haritası üretir.
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model (torch.nn.Module): Eğitilmiş CNN modeli
            target_layer (nn.Module): Görselleştirme yapılacak katman (örneğin: model.conv2)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None           # Backprop sırasında gradyanlar burada tutulacak
        self.activations = None         # Forward pass sırasında aktivasyonlar burada tutulacak

        # İlgili katmana forward ve backward hook bağlanıyor
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        """
        Forward pass sırasında çalışır.
        Katmanın çıkış aktivasyonlarını kaydeder.
        """
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        """
        Backward pass sırasında çalışır.
        Katmanın çıkışına göre gradyanları kaydeder.
        """
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x, target_class):
        """
        GradCAM ısı haritasını hesaplar.

        Args:
            x (torch.Tensor): Girdi görüntüsü (1 x C x H x W)
            target_class (int): Hedeflenen sınıf indeksi

        Returns:
            numpy.ndarray: Normalleştirilmiş 2D ısı haritası
        """
                # Modelin tahminini al
        output = self.model(x)

        # Hedef sınıfın loss'unu çıkar
        loss = output[:, target_class]

        # Geriye yayılım başlat
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # Kanal başına ortalama gradyan ağırlığı (Global Average Pooling)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Aktivasyonları gradyanlarla ağırlıklandır
        weighted_activations = weights * self.activations

        # Kanal boyunca toplayarak ısı haritasını üret
        heatmap = weighted_activations.sum(dim=1).squeeze()

        # Negatif değerleri sıfırla (ReLU)
        heatmap = F.relu(heatmap)

        # Normalize et (0-1 aralığına getir)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()
        
