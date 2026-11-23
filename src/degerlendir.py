"""Her kategori için yöntemleri eğitir, test eder, sonuçları ve grafikleri kaydeder.

Çalıştırma:
    python -m src.degerlendir                       # 15 kategori, tüm yöntemler
    python -m src.degerlendir --kategoriler bottle screw --yontemler patchcore
"""
import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .autoencoder import AutoencoderTespit
from .metrikler import aupro, goruntu_auroc, piksel_auroc
from .patchcore import OzellikCikarici, PatchCore
from .veri import KATEGORILER, KOK, MVTec, egitim_dogrulama_ayir, geri_cevir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_KLASORU = KOK / "models" / "patchcore"
RAPOR_KLASORU = KOK / "reports"
GRAFIK_KLASORU = RAPOR_KLASORU / "figures"


def yukleyici(veri, karistir=False):
    return DataLoader(veri, batch_size=16, shuffle=karistir, num_workers=0)


def topla(model, yuk):
    puanlar, haritalar = [], []
    for x, *_ in yuk:
        p, h = model.puanla(x)
        puanlar.append(p)
        haritalar.append(h)
    return np.concatenate(puanlar), np.concatenate(haritalar)


def kategori_calistir(kategori, cikarici, yontemler, ae_tur):
    egitim_tam, test = MVTec(kategori, "train"), MVTec(kategori, "test")
    egitim, dogrulama = egitim_dogrulama_ayir(egitim_tam)
    etiketler = np.array([k[1] for k in test.kayitlar])
    maskeler = np.concatenate([m.numpy() for _, _, m in yukleyici(test)])[:, 0].astype(bool)

    modeller = []
    if "patchcore" in yontemler:
        modeller.append(("PatchCore %1", PatchCore(cikarici, coreset_orani=0.01)))
        modeller.append(("PatchCore %10", PatchCore(cikarici, coreset_orani=0.10)))
    if "autoencoder" in yontemler:
        modeller.append(("Autoencoder", AutoencoderTespit(tur=ae_tur)))

    satirlar, haritalar = [], {}
    for ad, model in modeller:
        torch.manual_seed(0)
        bas = time.time()
        model.egit(yukleyici(egitim, karistir=ad == "Autoencoder"))
        egitim_suresi = time.time() - bas

        dogrulama_puani, dogrulama_haritasi = topla(model, yukleyici(dogrulama))
        esik = float(dogrulama_puani.max())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bas = time.time()
        puan, harita = topla(model, yukleyici(test))
        sure_ms = (time.time() - bas) / len(test) * 1000
        haritalar[ad] = harita

        tahmin = puan > esik
        satir = {
            "kategori": kategori,
            "yontem": ad,
            "goruntu_auroc": goruntu_auroc(etiketler, puan),
            "piksel_auroc": piksel_auroc(maskeler, harita),
            "aupro": aupro(maskeler, harita),
            "hata_yakalama": float(tahmin[etiketler == 1].mean()),
            "yanlis_alarm": float(tahmin[etiketler == 0].mean()),
            "egitim_sn": round(egitim_suresi, 1),
            "ms_goruntu": round(sure_ms, 1),
        }
        if isinstance(model, PatchCore):
            satir["bellek_mb"] = round(model.bellek.numel() * 2 / 2 ** 20, 1)  # float16 olarak
            if ad == "PatchCore %1":
                model.esik = esik
                # Harita yumuşatıldığı için değerleri görüntü puanından düşük kalıyor;
                # demo ve API'deki renklendirme için ayrı bir üst sınır saklıyorum
                model.harita_ust = float(dogrulama_haritasi.max())
                model.kaydet(MODEL_KLASORU / f"{kategori}.pt")
        satirlar.append(satir)
        print(f"  {ad:14s} görüntü {satir['goruntu_auroc']:.3f} | piksel {satir['piksel_auroc']:.3f} | "
              f"PRO {satir['aupro']:.3f} | eğitim {egitim_suresi:.0f} sn | {sure_ms:.0f} ms/görüntü", flush=True)

    ornekler_ciz(kategori, test, etiketler, maskeler, haritalar)
    return satirlar


def normalize(h, referans):
    alt, ust = np.percentile(referans, 1), np.percentile(referans, 99.9)
    return np.clip((h - alt) / (ust - alt + 1e-8), 0, 1)


def ornekler_ciz(kategori, test, etiketler, maskeler, haritalar, adet=4):
    # Hata alanı büyüklüğüne göre sıralayıp farklı boyutlardan örnekler seçiyorum
    hatali = np.flatnonzero(etiketler == 1)
    alan = maskeler[hatali].reshape(len(hatali), -1).sum(1)
    secim = hatali[np.argsort(alan)][np.linspace(0, len(hatali) - 1, adet).astype(int)]

    sutunlar = ["Görüntü", "Gerçek hata"] + list(haritalar)
    fig, eksenler = plt.subplots(adet, len(sutunlar), figsize=(2.6 * len(sutunlar), 2.6 * adet))
    for satir, i in zip(eksenler, secim):
        goruntu = geri_cevir(test[i][0])
        satir[0].imshow(goruntu)
        satir[1].imshow(goruntu)
        satir[1].contour(maskeler[i], colors="red", linewidths=1)
        for eksen, (ad, h) in zip(satir[2:], haritalar.items()):
            eksen.imshow(goruntu)
            eksen.imshow(normalize(h[i], h), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        satir[0].set_ylabel(test.kayitlar[i][3], fontsize=9)
    for eksen, ad in zip(eksenler[0], sutunlar):
        eksen.set_title(ad, fontsize=10)
    for eksen in eksenler.ravel():
        eksen.set_xticks([])
        eksen.set_yticks([])
    fig.suptitle(kategori)
    plt.tight_layout()
    (GRAFIK_KLASORU / "kategoriler").mkdir(parents=True, exist_ok=True)
    plt.savefig(GRAFIK_KLASORU / "kategoriler" / f"{kategori}.jpg", dpi=110, pil_kwargs={"quality": 85})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kategoriler", nargs="+", default=KATEGORILER)
    ap.add_argument("--yontemler", nargs="+", default=["patchcore", "autoencoder"])
    ap.add_argument("--ae-tur", type=int, default=150)
    ap.add_argument("--cikti", default="sonuclar.csv")
    args = ap.parse_args()

    for k in (MODEL_KLASORU, GRAFIK_KLASORU):
        k.mkdir(parents=True, exist_ok=True)
    cikarici = OzellikCikarici()

    satirlar = []
    for kategori in args.kategoriler:
        print(kategori, flush=True)
        satirlar += kategori_calistir(kategori, cikarici, args.yontemler, args.ae_tur)
        pd.DataFrame(satirlar).to_csv(RAPOR_KLASORU / args.cikti, index=False, float_format="%.4f")

    tablo = pd.DataFrame(satirlar)
    ozet = tablo.groupby("yontem")[["goruntu_auroc", "piksel_auroc", "aupro", "hata_yakalama",
                                    "yanlis_alarm", "ms_goruntu"]].mean().round(4)
    print("\nOrtalama:\n", ozet.to_string())
    if args.kategoriler == KATEGORILER:
        with open(RAPOR_KLASORU / "ozet.json", "w", encoding="utf-8") as f:
            json.dump(ozet.to_dict(orient="index"), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
