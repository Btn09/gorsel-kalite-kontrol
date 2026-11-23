"""Karşılaştırma için klasik yöntem: sağlam görüntülerle eğitilen bir autoencoder.
Hatalı bölgeyi iyi geri üretemeyeceği varsayılıyor, geri üretme hatası hata haritası oluyor."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn


def blok(gir, cik):
    return nn.Sequential(nn.Conv2d(gir, cik, 4, 2, 1), nn.BatchNorm2d(cik), nn.LeakyReLU(0.2))


def ters_blok(gir, cik):
    return nn.Sequential(nn.ConvTranspose2d(gir, cik, 4, 2, 1), nn.BatchNorm2d(cik), nn.ReLU())


class Autoencoder(nn.Module):
    def __init__(self, darbogaz=64):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.kodlayici = nn.Sequential(
            blok(3, 32), blok(32, 64), blok(64, 128), blok(128, 256), blok(256, 256),
            nn.Conv2d(256, darbogaz, 1),
        )
        self.cozucu = nn.Sequential(
            nn.Conv2d(darbogaz, 256, 1), nn.ReLU(),
            ters_blok(256, 256), ters_blok(256, 128), ters_blok(128, 64), ters_blok(64, 32),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x):
        return self.cozucu(self.kodlayici(x))


class AutoencoderTespit:
    def __init__(self, cihaz=None, tur=150, ogrenme_orani=1e-3):
        self.cihaz = cihaz or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder().to(self.cihaz)
        self.tur = tur
        self.ogrenme_orani = ogrenme_orani

    def egit(self, yukleyici, grup=16):
        # Görüntüleri bir kere okuyup ekran kartında tutuyorum; her turda diskten PNG okumak
        # eğitimi 10 kat yavaşlatıyordu
        veri = torch.cat([x for x, *_ in yukleyici]).to(self.cihaz)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.ogrenme_orani)
        zamanlayici = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.tur)
        self.model.train()
        for _ in range(self.tur):
            for sira in torch.randperm(len(veri), device=self.cihaz).split(grup):
                x = veri[sira]
                kayip = nn.functional.mse_loss(self.model(x), x)
                opt.zero_grad()
                kayip.backward()
                opt.step()
            zamanlayici.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def puanla(self, x):
        x = x.to(self.cihaz)
        hata = ((self.model(x) - x) ** 2).mean(1).cpu().numpy()
        harita = np.stack([gaussian_filter(m, sigma=4) for m in hata])
        return harita.reshape(len(harita), -1).max(1), harita
