"""Görsel kalite kontrol servisi.

Çalıştırma:  uvicorn api.app:app
Dokümantasyon: http://127.0.0.1:8000/docs
"""
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from src.cikarim import KaliteKontrol, png_bayt

durum = {}


@asynccontextmanager
async def lifespan(app):
    durum["kk"] = KaliteKontrol()
    yield
    durum.clear()


app = FastAPI(title="Görsel Kalite Kontrol", lifespan=lifespan)


def goruntu_oku(dosya):
    try:
        return Image.open(io.BytesIO(dosya.file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Dosya bir görüntü değil")


def kategori_kontrol(kategori):
    if kategori not in durum["kk"].kategoriler:
        raise HTTPException(404, f"Bilinmeyen kategori: {kategori}")


@app.get("/kategoriler")
def kategoriler():
    return durum["kk"].kategoriler


@app.post("/kontrol/{kategori}")
def kontrol(kategori: str, dosya: UploadFile = File(...)):
    kategori_kontrol(kategori)
    sonuc, _, _ = durum["kk"].kontrol_et(goruntu_oku(dosya), kategori)
    return sonuc


@app.post("/kontrol/{kategori}/harita", response_class=Response)
def harita(kategori: str, dosya: UploadFile = File(...)):
    """Hata haritasını görüntünün üstüne bindirilmiş PNG olarak döndürür."""
    kategori_kontrol(kategori)
    sonuc, x, h = durum["kk"].kontrol_et(goruntu_oku(dosya), kategori)
    ust = durum["kk"].ust_uste_bindir(x, h, kategori)
    return Response(png_bayt(ust), media_type="image/png",
                    headers={"X-Hatali": str(sonuc["hatali"]), "X-Puan": str(sonuc["puan"])})
