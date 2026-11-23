"""Tarayıcıda görüntü yükleyip sonucu görmek için küçük bir arayüz.

Çalıştırma:  python demo.py   ->  http://127.0.0.1:7860
"""
from pathlib import Path

import gradio as gr

from src.cikarim import KaliteKontrol

kk = KaliteKontrol()
ORNEK_KLASORU = Path(__file__).parent / "ornekler"


def kontrol(goruntu, kategori):
    if goruntu is None:
        return None, "Önce bir görüntü yükleyin."
    sonuc, x, harita = kk.kontrol_et(goruntu, kategori)
    karar = "HATALI" if sonuc["hatali"] else "SAĞLAM"
    metin = (f"**{karar}**  \nPuan: {sonuc['puan']} (eşik {sonuc['esik']})  \n"
             f"Eşiği geçen alan: %{sonuc['hata_alani_orani'] * 100:.2f}")
    return kk.ust_uste_bindir(x, harita, kategori), metin


ornekler = [[str(p), p.name.split("__")[0]] for p in sorted(ORNEK_KLASORU.glob("*.png"))]

with gr.Blocks(title="Görsel Kalite Kontrol") as arayuz:
    gr.Markdown("## Görsel Kalite Kontrol\nÜrün türünü seçip bir fotoğraf yükleyin. "
                "Model yalnızca sağlam ürünlerle hazırlandı; normalden farklı bölgeler kırmızı görünür.")
    with gr.Row():
        with gr.Column():
            giris = gr.Image(type="pil", label="Ürün görüntüsü")
            kategori = gr.Dropdown(kk.kategoriler, value=kk.kategoriler[0], label="Ürün türü")
            dugme = gr.Button("Kontrol et", variant="primary")
        with gr.Column():
            cikis = gr.Image(label="Hata haritası")
            sonuc = gr.Markdown()
    dugme.click(kontrol, [giris, kategori], [cikis, sonuc])
    if ornekler:
        gr.Examples(ornekler, [giris, kategori])

if __name__ == "__main__":
    arayuz.launch()
