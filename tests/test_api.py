"""
API tests — run with: pytest tests/ -v

The EfficientNet model is a module-level singleton so it loads once per
process (~5 s on first run) and is reused across all tests.
"""
import io
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from app.main import app


def synthetic_image(width: int = 64, height: int = 64, fmt: str = "JPEG") -> bytes:
    """Return bytes of a tiny solid-colour image — no disk I/O needed."""
    img = Image.new("RGB", (width, height), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── Health / root ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root():
    async with await _client() as ac:
        r = await ac.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


@pytest.mark.asyncio
async def test_health():
    async with await _client() as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "version" in body


# ── Predict — happy path ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_jpeg_returns_top5():
    """A valid JPEG → 200 with 5 predictions, a label string, and latency."""
    async with await _client() as ac:
        r = await ac.post(
            "/predict",
            files={"file": ("test.jpg", synthetic_image(), "image/jpeg")},
        )
    assert r.status_code == 200
    body = r.json()

    assert len(body["predictions"]) == 5
    assert body["inference_ms"] > 0

    top = body["predictions"][0]
    assert isinstance(top["label"], str) and top["label"]
    assert 0.0 <= top["confidence"] <= 100.0


@pytest.mark.asyncio
async def test_predict_png():
    async with await _client() as ac:
        r = await ac.post(
            "/predict",
            files={"file": ("test.png", synthetic_image(fmt="PNG"), "image/png")},
        )
    assert r.status_code == 200


# ── Predict — error cases ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_no_file_returns_422():
    async with await _client() as ac:
        r = await ac.post("/predict")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_non_image_returns_415():
    async with await _client() as ac:
        r = await ac.post(
            "/predict",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
    assert r.status_code == 415


@pytest.mark.asyncio
async def test_predict_oversized_file_returns_413():
    big = b"x" * (11 * 1024 * 1024)  # 11 MB
    async with await _client() as ac:
        r = await ac.post(
            "/predict",
            files={"file": ("big.jpg", big, "image/jpeg")},
        )
    assert r.status_code == 413


@pytest.mark.asyncio
async def test_predict_corrupt_image_returns_422():
    async with await _client() as ac:
        r = await ac.post(
            "/predict",
            files={"file": ("bad.jpg", b"not-an-image-at-all", "image/jpeg")},
        )
    assert r.status_code == 422
