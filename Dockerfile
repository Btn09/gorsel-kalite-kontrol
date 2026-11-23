FROM python:3.12-slim

WORKDIR /app
# Ekran kartı olmadan çalışan (CPU) PyTorch, imaj daha küçük oluyor
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Önceden eğitilmiş ağırlıkları imaja koy, açılışta internet gerekmesin
RUN python -c "from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights as W; wide_resnet50_2(weights=W.IMAGENET1K_V1)"

COPY src/ src/
COPY api/ api/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
