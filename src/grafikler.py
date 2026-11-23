"""reports/sonuclar.csv'den özet grafikleri üretir.  Çalıştırma: python -m src.grafikler"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .veri import KOK  # noqa: E402

RAPOR = KOK / "reports"
RENKLER = {"PatchCore %1": "tab:blue", "PatchCore %10": "tab:cyan", "Autoencoder": "tab:orange"}


def main():
    tablo = pd.read_csv(RAPOR / "sonuclar.csv")
    yontemler = [y for y in RENKLER if y in set(tablo.yontem)]
    kategoriler = list(dict.fromkeys(tablo.kategori))
    x = np.arange(len(kategoriler))
    genislik = 0.8 / len(yontemler)

    fig, eksenler = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for eksen, (sutun, baslik) in zip(eksenler, [("goruntu_auroc", "Görüntü AUROC (ürün hatalı mı?)"),
                                                  ("aupro", "AUPRO (hata nerede?)")]):
        for i, y in enumerate(yontemler):
            d = tablo[tablo.yontem == y].set_index("kategori").loc[kategoriler, sutun]
            eksen.bar(x + (i - (len(yontemler) - 1) / 2) * genislik, d, genislik,
                      label=f"{y} (ort. {d.mean():.3f})", color=RENKLER[y])
        eksen.set_title(baslik)
        eksen.set_ylim(0.3, 1.12)
        eksen.grid(axis="y", alpha=0.3)
        eksen.legend(loc="upper center", ncol=len(yontemler), fontsize=8)
    eksenler[1].set_xticks(x, kategoriler, rotation=30)
    plt.tight_layout()
    plt.savefig(RAPOR / "figures" / "kategori_karsilastirma.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
