"""Kayıtlı PatchCore modelleriyle tek görüntü kontrolü. API ve demo bunu kullanıyor."""
import io
import os
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

from .patchcore import OzellikCikarici, PatchCore
from .veri import KOK, geri_cevir, goruntu_hazirla

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

MODEL_KLASORU = Path(os.environ.get("MODEL_KLASORU", KOK / "models" / "patchcore"))


class KaliteKontrol:
    def __init__(self, model_klasoru=MODEL_KLASORU):
        cikarici = OzellikCikarici()
        self.modeller = {
            yol.stem: PatchCore(cikarici).yukle(yol)
            for yol in sorted(Path(model_klasoru).glob("*.pt"))
        }
        if not self.modeller:
            raise FileNotFoundError(f"{model_klasoru} içinde model yok. Önce python -m src.degerlendir çalıştırın.")

    @property
    def kategoriler(self):
        return list(self.modeller)

    def kontrol_et(self, goruntu, kategori):
        model = self.modeller[kategori]
        x = goruntu_hazirla(goruntu)
        puan, harita = model.puanla(x)
        puan, harita = float(puan[0]), harita[0]
        # Haritada, sağlam ürünlerde hiç görülmeyen değerlere çıkan piksellerin oranı:
        # hatanın ne kadar alan kapladığına dair kaba bir ölçü
        return {
            "kategori": kategori,
            "puan": round(puan, 3),
            "esik": round(model.esik, 3),
            "hatali": puan > model.esik,
            "hata_alani_orani": round(float((harita > model.harita_ust).mean()), 4),
        }, x[0], harita

    def ust_uste_bindir(self, x, harita, kategori):
        goruntu = geri_cevir(x)
        # Renk ölçeği sağlam ürünlerdeki en yüksek harita değerine göre:
        # %75'inin altı mavi, %125'i ve üstü kırmızı
        ust = self.modeller[kategori].harita_ust
        oran = np.clip((harita - 0.75 * ust) / (0.5 * ust), 0, 1)
        renkli = colormaps["jet"](oran)[..., :3]
        karisim = goruntu * 0.55 + renkli * 0.45
        return Image.fromarray((karisim * 255).astype(np.uint8))


def png_bayt(goruntu):
    tampon = io.BytesIO()
    goruntu.save(tampon, format="PNG")
    return tampon.getvalue()
