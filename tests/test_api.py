from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app

ORNEKLER = Path(__file__).resolve().parent.parent / "ornekler"


def ornek(ad):
    return {"dosya": (ad, (ORNEKLER / ad).read_bytes(), "image/png")}


@pytest.fixture(scope="module")
def istemci():
    with TestClient(app) as c:
        yield c


def test_kategoriler(istemci):
    kategoriler = istemci.get("/kategoriler").json()
    assert len(kategoriler) == 15
    assert "bottle" in kategoriler


@pytest.mark.parametrize("ad, beklenen", [
    ("bottle__saglam.png", False),
    ("bottle__broken_large.png", True),
    ("screw__saglam.png", False),
    ("screw__scratch_head.png", True),
])
def test_kontrol(istemci, ad, beklenen):
    kategori = ad.split("__")[0]
    sonuc = istemci.post(f"/kontrol/{kategori}", files=ornek(ad)).json()
    assert sonuc["hatali"] is beklenen


def test_harita_png(istemci):
    cevap = istemci.post("/kontrol/bottle/harita", files=ornek("bottle__broken_large.png"))
    assert cevap.status_code == 200
    assert cevap.headers["content-type"] == "image/png"
    assert cevap.content[:4] == b"\x89PNG"


def test_bilinmeyen_kategori(istemci):
    assert istemci.post("/kontrol/araba", files=ornek("bottle__saglam.png")).status_code == 404


def test_goruntu_olmayan_dosya(istemci):
    dosya = {"dosya": ("not.txt", b"merhaba", "text/plain")}
    assert istemci.post("/kontrol/bottle", files=dosya).status_code == 400
