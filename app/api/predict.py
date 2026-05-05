from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
from app.core.model import model_manager

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run image classification inference"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = model_manager.predict(image)
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load-model")
async def load_model():
    """Load model from S3"""
    try:
        model_manager.load_from_s3()
        return {"status": "Model loaded from S3 successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))