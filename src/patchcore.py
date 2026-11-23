"""PatchCore (Roth ve ark., 2022): sağlam ürünlerin yama özelliklerinden bir bellek oluşturup
test görüntüsündeki her yamanın bu belleğe uzaklığını hata puanı olarak kullanır. Eğitim yok,
sadece önceden eğitilmiş bir ağdan özellik çıkarılıyor."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

OZELLIK_BOYUTU = 1024


class OzellikCikarici(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ag = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.govde = torch.nn.Sequential(ag.conv1, ag.bn1, ag.relu, ag.maxpool, ag.layer1)
        self.katman2, self.katman3 = ag.layer2, ag.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        # Orta katmanları kullanıyorum: ilk katmanlar fazla genel, son katmanlar ImageNet sınıflarına fazla özel
        k2 = self.katman2(self.govde(x))   # 512 x 28 x 28
        k3 = self.katman3(k2)              # 1024 x 14 x 14
        # Her yamaya komşularını da kat (3x3 ortalama), sonra iki katmanı aynı çözünürlükte birleştir
        k2 = F.avg_pool2d(k2, 3, 1, 1)
        k3 = F.avg_pool2d(k3, 3, 1, 1)
        k3 = F.interpolate(k3, size=k2.shape[-2:], mode="bilinear", align_corners=False)
        oz = torch.cat([k2, k3], 1)
        b, c, h, w = oz.shape
        oz = oz.permute(0, 2, 3, 1).reshape(-1, 1, c)
        oz = F.adaptive_avg_pool1d(oz, OZELLIK_BOYUTU).squeeze(1)
        return oz, (b, h, w)


def coreset_sec(oz, oran, seed=0):
    """Açgözlü k-merkez seçimi: belleği küçültürken özellik uzayını mümkün olduğunca iyi kapsayan
    yamaları seçer. Hız için uzaklıklar 128 boyuta rastgele izdüşümle hesaplanıyor."""
    n = len(oz)
    k = max(1, int(n * oran))
    g = torch.Generator(device=oz.device).manual_seed(seed)
    izdusum = torch.randn(oz.shape[1], 128, device=oz.device, generator=g) / 128 ** 0.5
    z = oz @ izdusum
    secilen = [int(torch.randint(n, (1,), device=oz.device, generator=g))]
    en_yakin = torch.cdist(z, z[secilen]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(en_yakin))
        secilen.append(i)
        en_yakin = torch.minimum(en_yakin, torch.cdist(z, z[i:i + 1]).squeeze(1))
    return oz[secilen]


class PatchCore:
    def __init__(self, cikarici=None, cihaz=None, coreset_orani=0.01):
        self.cihaz = cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cikarici = (cikarici or OzellikCikarici()).to(self.cihaz)
        self.coreset_orani = coreset_orani
        self.bellek = None
        self.esik = None          # görüntü puanı için karar eşiği
        self.harita_ust = None    # sağlam ürünlerde hata haritasının ulaştığı en yüksek değer

    def egit(self, yukleyici):
        parcalar = []
        for x, *_ in yukleyici:
            oz, _ = self.cikarici(x.to(self.cihaz))
            parcalar.append(oz)
        tum = torch.cat(parcalar)
        self.bellek = coreset_sec(tum, self.coreset_orani)
        return self

    @torch.no_grad()
    def puanla(self, x):
        """Görüntü puanları (B,) ve 224x224 hata haritaları (B, H, W) döndürür."""
        oz, (b, h, w) = self.cikarici(x.to(self.cihaz))
        bellek = self.bellek.to(self.cihaz, torch.float32)
        mesafe = torch.cat([torch.cdist(p, bellek).min(1).values for p in oz.split(8192)])
        yama = mesafe.reshape(b, 1, h, w)
        goruntu_puani = yama.flatten(1).max(1).values
        harita = F.interpolate(yama, size=x.shape[-2:], mode="bilinear", align_corners=False)
        harita = np.stack([gaussian_filter(m, sigma=4) for m in harita[:, 0].cpu().numpy()])
        return goruntu_puani.cpu().numpy(), harita

    def kaydet(self, yol):
        torch.save({"bellek": self.bellek.half().cpu(), "esik": self.esik,
                    "harita_ust": self.harita_ust, "coreset_orani": self.coreset_orani}, yol)

    def yukle(self, yol):
        d = torch.load(yol, map_location="cpu")
        self.bellek = d["bellek"].float().to(self.cihaz)
        self.esik = d["esik"]
        self.harita_ust = d.get("harita_ust", d["esik"])
        self.coreset_orani = d["coreset_orani"]
        return self
