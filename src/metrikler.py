import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def goruntu_auroc(etiketler, puanlar):
    return roc_auc_score(etiketler, puanlar)


def piksel_auroc(maskeler, haritalar):
    return roc_auc_score(maskeler.ravel().astype(np.uint8), haritalar.ravel())


def aupro(maskeler, haritalar, fpr_siniri=0.3, adim=1000):
    """Per-Region Overlap eğrisinin altındaki alan (yanlış pozitif oranı 0-0.3 arası, normalize).
    Piksel AUROC büyük hataları fazla ödüllendiriyor; PRO her hatalı bölgeyi eşit sayıyor,
    küçük bir çizik de büyük bir kırık kadar önemli oluyor."""
    maskeler = maskeler.astype(bool)
    esikler = np.linspace(haritalar.max(), haritalar.min(), adim)

    normal = np.sort(haritalar[~maskeler])
    fpr = 1 - np.searchsorted(normal, esikler, side="left") / len(normal)

    pro_toplam = np.zeros(adim)
    bolge_sayisi = 0
    for m, h in zip(maskeler, haritalar):
        etiket, n = ndimage.label(m)
        for r in range(1, n + 1):
            s = np.sort(h[etiket == r])
            pro_toplam += 1 - np.searchsorted(s, esikler, side="left") / len(s)
            bolge_sayisi += 1
    pro = pro_toplam / bolge_sayisi

    secili = fpr <= fpr_siniri
    x = np.append(fpr[secili], fpr_siniri)
    y = np.append(pro[secili], np.interp(fpr_siniri, fpr, pro))
    return float(np.trapezoid(y, x) / fpr_siniri)
