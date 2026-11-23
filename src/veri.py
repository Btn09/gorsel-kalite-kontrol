from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T

KOK = Path(__file__).resolve().parent.parent
VERI_KLASORU = KOK / "data" / "mvtec_ad"

KATEGORILER = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
DOKULAR = {"carpet", "grid", "leather", "tile", "wood"}

BOYUT, KIRPMA = 256, 224
ORTALAMA, SAPMA = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

goruntu_donustur = T.Compose([
    T.Resize((BOYUT, BOYUT)),
    T.CenterCrop(KIRPMA),
    T.ToTensor(),
    T.Normalize(ORTALAMA, SAPMA),
])
maske_donustur = T.Compose([
    T.Resize((BOYUT, BOYUT), interpolation=T.InterpolationMode.NEAREST),
    T.CenterCrop(KIRPMA),
    T.ToTensor(),
])


def goruntu_hazirla(goruntu):
    """PIL görüntüsünü modele verilecek tensöre çevirir (1, 3, 224, 224)."""
    return goruntu_donustur(goruntu.convert("RGB")).unsqueeze(0)


def geri_cevir(x):
    """Normalize edilmiş tensörü gösterilebilir (H, W, 3) diziye çevirir."""
    x = x.cpu().numpy().transpose(1, 2, 0) * SAPMA + ORTALAMA
    return np.clip(x, 0, 1)


class MVTec(Dataset):
    def __init__(self, kategori, bolum, kok=VERI_KLASORU):
        kok = Path(kok)
        klasor = kok / kategori / bolum
        if not klasor.exists():
            raise FileNotFoundError(f"{klasor} yok. MVTec AD'yi indirip data/mvtec_ad/ altına açın.")
        self.kayitlar = []  # (görüntü yolu, etiket, maske yolu, hata türü)
        for tur in sorted(p for p in klasor.iterdir() if p.is_dir()):
            for yol in sorted(tur.glob("*.png")):
                if tur.name == "good":
                    self.kayitlar.append((yol, 0, None, "good"))
                else:
                    maske = kok / kategori / "ground_truth" / tur.name / f"{yol.stem}_mask.png"
                    self.kayitlar.append((yol, 1, maske, tur.name))

    def __len__(self):
        return len(self.kayitlar)

    def __getitem__(self, i):
        yol, etiket, maske_yolu, _ = self.kayitlar[i]
        x = goruntu_donustur(Image.open(yol).convert("RGB"))
        if maske_yolu is None:
            m = torch.zeros(1, KIRPMA, KIRPMA)
        else:
            m = (maske_donustur(Image.open(maske_yolu).convert("L")) > 0.5).float()
        return x, etiket, m


def egitim_dogrulama_ayir(veri, oran=0.1, seed=0):
    # Eğitimdeki sağlam görüntülerin %10'unu eşik seçimi için ayırıyorum.
    # Fabrikada hatalı örnek az olduğu için eşik sadece sağlam ürünlere bakılarak seçiliyor.
    sira = np.random.default_rng(seed).permutation(len(veri))
    n = max(1, int(len(veri) * oran))
    return Subset(veri, sira[n:].tolist()), Subset(veri, sira[:n].tolist())
