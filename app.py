"""
Artisera AI Image Enhancement Microservice
==========================================
A production-ready FastAPI service for artisan product background segmentation,
mask refinement, clean ecommerce compositing, and conservative enhancement.

Architecture:
  Flutter / Node.js -> POST /enhance (multipart/form-data: images) -> BiRefNet-HR -> Mask Refine -> Clean Studio Output
"""

import io
import os
import time
import base64
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForImageSegmentation

# ==============================================================================
# 1. LOGGING & CONFIGURATION
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("artisera-image-ai")

# Model Configuration
MODEL_ID = "ZhengPeng7/BiRefNet_HR"

# Maximum payload limits
MAX_IMAGE_SIZE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_BYTES", 25 * 1024 * 1024))  # 25 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# CORS Configuration
# In production, specify explicit domain origins e.g. "https://artisera.app,https://admin.artisera.app"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Runtime state container
state = {
    "model": None,
    "device": "cpu",
    "use_fp16": False,
    "transform": None,
    "gpu_name": None,
    "is_ready": False,
}

# ==============================================================================
# 2. MODEL LOADING & LIFESPAN
# ==============================================================================

def init_model():
    """
    Initializes and loads BiRefNet-HR weights onto the optimal device.
    Uses CUDA with FP16 when available, otherwise falls back gracefully to CPU.
    """
    logger.info("[Artisera] Detecting compute hardware...")
    
    if torch.cuda.is_available():
        device = "cuda"
        use_fp16 = True
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"[Artisera] CUDA device detected: {gpu_name} (Using FP16)")
    else:
        device = "cpu"
        use_fp16 = False
        gpu_name = None
        logger.info("[Artisera] No CUDA GPU detected. Running on CPU (Float32)")

    logger.info(f"[Artisera] Loading model '{MODEL_ID}'...")
    try:
        if use_fp16:
            model = AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )
        
        model.to(device)
        model.eval()
        
        state["model"] = model
        state["device"] = device
        state["use_fp16"] = use_fp16
        state["gpu_name"] = gpu_name
        state["is_ready"] = True
        logger.info(f"[Artisera] Model '{MODEL_ID}' successfully loaded into memory on {device.upper()}.")
    except Exception as e:
        logger.error(f"[Artisera] Failed to load model {MODEL_ID}: {str(e)}", exc_info=True)
        state["is_ready"] = False
        raise RuntimeError(f"Model initialization failed: {e}")

    # Standard BiRefNet-HR image preprocessing transform
    state["transform"] = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load the AI model once at startup."""
    init_model()
    yield
    # Cleanup resources on shutdown if needed
    if state["device"] == "cuda":
        torch.cuda.empty_cache()
    logger.info("[Artisera] Microservice shutdown complete.")


# ==============================================================================
# 3. FASTAPI APP & CORS
# ==============================================================================

app = FastAPI(
    title="Artisera AI Image Enhancement Service",
    description="High-resolution background removal and conservative enhancement for artisan products",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware configuration
# NOTE FOR PRODUCTION: Restrict CORS_ORIGINS to trusted internal domains or mobile gateways.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 4. IMAGE PREPROCESSING & INFERENCE
# ==============================================================================

def preprocess_image_for_birefnet(image: Image.Image, device: str, use_fp16: bool) -> torch.Tensor:
    """Prepares PIL RGB image into a normalized tensor ready for BiRefNet-HR inference."""
    transform = state["transform"]
    tensor = transform(image).unsqueeze(0).to(device)
    if use_fp16:
        tensor = tensor.half()
    return tensor


def run_segmentation_inference(image: Image.Image) -> np.ndarray:
    """
    Executes BiRefNet-HR inference to produce a high-resolution foreground probability mask.
    Returns:
        np.ndarray: 2D float32 array in range [0.0, 1.0] matching the original image dimensions.
    """
    model = state["model"]
    device = state["device"]
    use_fp16 = state["use_fp16"]
    
    if model is None or not state["is_ready"]:
        raise RuntimeError("Segmentation model is not ready.")

    orig_w, orig_h = image.size
    input_tensor = preprocess_image_for_birefnet(image, device, use_fp16)

    with torch.inference_mode():
        preds = model(input_tensor)
        
        # BiRefNet returns a list/tuple of multi-scale outputs; the last item is the fine prediction
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]
        else:
            pred = preds
            
        # Apply sigmoid to extract foreground probabilities
        pred = torch.sigmoid(pred)
        mask_tensor = pred.squeeze().float().cpu()
        mask_np = mask_tensor.numpy()

    # Free device memory for sequential safety
    del input_tensor, preds, pred
    if device == "cuda":
        torch.cuda.empty_cache()

    # Resize mask back to original image dimensions with smooth bilinear interpolation
    mask_full = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_full = np.clip(mask_full, 0.0, 1.0)
    return mask_full


# ==============================================================================
# 5. MASK REFINEMENT
# ==============================================================================

def refine_product_mask(mask: np.ndarray) -> np.ndarray:
    """
    Applies conservative morphological edge cleaning and anti-aliased Gaussian smoothing.
    Preserves exact artisan boundaries, decorations, embroidery, and fringe details.
    
    Args:
        mask (np.ndarray): 2D float array in [0.0, 1.0]
    Returns:
        np.ndarray: Refined 2D uint8 mask in [0, 255]
    """
    # Convert to 8-bit representation
    mask_u8 = (mask * 255.0).astype(np.uint8)
    
    # Small 3x3 elliptical kernel to remove subtle isolated speckle noise without eroding product edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Subtle Gaussian smoothing for anti-aliased edge blending
    mask_blurred = cv2.GaussianBlur(mask_cleaned, (3, 3), 0)
    
    return mask_blurred


# ==============================================================================
# 6. COMPOSITING & CONSERVATIVE IMAGE ENHANCEMENT
# ==============================================================================

def apply_conservative_enhancement(image: Image.Image) -> Image.Image:
    """
    Applies non-destructive, conservative optical enhancement:
    - Subtle contrast & brightness normalization
    - Gentle micro-contrast / sharpness boost
    - Preserves true artisan colors, textures, patterns, and embroidery without hallucination.
    """
    # Step 1: Subtle Brightness Polish (+3%)
    enhancer_b = ImageEnhance.Brightness(image)
    image = enhancer_b.enhance(1.03)

    # Step 2: Gentle Contrast Polish (+5%)
    enhancer_c = ImageEnhance.Contrast(image)
    image = enhancer_c.enhance(1.05)

    # Step 3: Natural Color Vibrancy Check (+2% natural warmth preservation)
    enhancer_col = ImageEnhance.Color(image)
    image = enhancer_col.enhance(1.02)

    # Step 4: Crisp Edge Micro-Sharpness (+10% clarity)
    enhancer_s = ImageEnhance.Sharpness(image)
    image = enhancer_s.enhance(1.10)

    return image


def composite_on_white_background(orig_image: Image.Image, refined_mask: np.ndarray) -> Image.Image:
    """
    Isolates the artisan product with the refined alpha mask and composites it onto
    a clean, pure white (#FFFFFF) ecommerce studio canvas.
    """
    orig_w, orig_h = orig_image.size
    
    # Create RGBA product cutout
    r, g, b = orig_image.split()
    alpha_img = Image.fromarray(refined_mask, mode="L")
    rgba_product = Image.merge("RGBA", (r, g, b, alpha_img))
    
    # Create studio pure-white background
    white_bg = Image.new("RGBA", (orig_w, orig_h), (255, 255, 255, 255))
    
    # Clean alpha composite
    composited = Image.alpha_composite(white_bg, rgba_product)
    final_rgb = composited.convert("RGB")
    
    # Apply conservative enhancement
    enhanced_final = apply_conservative_enhancement(final_rgb)
    return enhanced_final


def process_single_image(image_bytes: bytes, filename: str) -> dict:
    """
    End-to-end processing pipeline for a single artisan product image:
    1. Read and validate
    2. Segregate foreground via BiRefNet-HR
    3. Refine boundary mask
    4. Composite onto #FFFFFF canvas
    5. Conservative enhancement
    6. Return encoded base64 PNG
    """
    t_start = time.time()
    logger.info(f"[Artisera] Processing image: {filename}")
    
    try:
        # STEP 1: Read image via Pillow
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        # Handle EXIF orientation if present
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # STEP 2 & 3: Convert to RGB and preserve original
        orig_image = pil_img.convert("RGB")
    except Exception as e:
        logger.error(f"[Artisera] Failed to decode image '{filename}': {str(e)}")
        raise ValueError(f"Invalid or corrupted image format for {filename}")

    # STEP 4, 5, 6, 7: BiRefNet-HR inference and mask generation
    t_seg_start = time.time()
    raw_mask = run_segmentation_inference(orig_image)
    t_seg = time.time() - t_seg_start

    # STEP 8, 9, 10, 11, 12, 13, 14: Mask refinement & White canvas compositing
    t_enh_start = time.time()
    refined_mask = refine_product_mask(raw_mask)
    final_image = composite_on_white_background(orig_image, refined_mask)
    t_enh = time.time() - t_enh_start

    # Export to memory as PNG
    output_buffer = io.BytesIO()
    final_image.save(output_buffer, format="PNG", optimize=True)
    png_bytes = output_buffer.getvalue()
    b64_str = base64.b64encode(png_bytes).decode("utf-8")

    t_total = time.time() - t_start
    logger.info(f"[Artisera] Segmentation: {t_seg:.2f}s | Enhancement: {t_enh:.2f}s | Total: {t_total:.2f}s for {filename}")

    return {
        "filename": filename,
        "image_base64": b64_str,
        "content_type": "image/png",
        "processing_time_sec": round(t_total, 3),
        "dimensions": {
            "width": orig_image.width,
            "height": orig_image.height,
        }
    }


# ==============================================================================
# 7. PYDANTIC RESPONSE SCHEMAS
# ==============================================================================

class ProcessedImageItem(BaseModel):
    filename: str
    image_base64: str
    content_type: str = "image/png"


class EnhanceResponse(BaseModel):
    success: bool
    count: int
    images: List[ProcessedImageItem]


class HealthResponse(BaseModel):
    status: str
    service: str


class GpuResponse(BaseModel):
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    model: str


# ==============================================================================
# 8. API ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Lightweight health check endpoint for container orchestrators and Flutter/Node ping."""
    return {
        "status": "ok" if state["is_ready"] else "loading",
        "service": "artisera-image-ai",
    }


@app.get("/gpu", response_model=GpuResponse, tags=["Monitoring"])
async def gpu_status():
    """Returns hardware acceleration details, CUDA state, device name, and loaded model ID."""
    return {
        "device": state["device"],
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": state["gpu_name"],
        "model": MODEL_ID,
    }


@app.post("/enhance", response_model=EnhanceResponse, tags=["Enhancement"])
async def enhance_images(images: List[UploadFile] = File(...)):
    """
    Main AI Image Enhancement endpoint.
    Accepts multipart/form-data containing one or multiple artisan product images under 'images'.
    
    Returns clean ecommerce cutout on #FFFFFF background with conservative enhancement.
    """
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided in the request.")

    if not state["is_ready"]:
        raise HTTPException(status_code=503, detail="AI Model is still initializing. Please retry in a few seconds.")

    results: List[ProcessedImageItem] = []

    for upload_file in images:
        filename = os.path.basename(upload_file.filename or "product.jpg")
        
        # Extension & MIME safety checks
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS and (upload_file.content_type not in ALLOWED_MIME_TYPES if upload_file.content_type else False):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for '{filename}'. Allowed formats: JPG, JPEG, PNG, WEBP."
            )

        try:
            # Read file bytes into memory
            content = await upload_file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail=f"Uploaded file '{filename}' is empty.")
            
            if len(content) > MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{filename}' exceeds maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB."
                )

            # Process sequentially to prevent VRAM over-allocation
            processed_data = process_single_image(content, filename)
            
            results.append(
                ProcessedImageItem(
                    filename=processed_data["filename"],
                    image_base64=processed_data["image_base64"],
                    content_type=processed_data["content_type"],
                )
            )
        except HTTPException:
            raise
        except ValueError as val_err:
            logger.warning(f"[Artisera] Image decoding failure for '{filename}': {str(val_err)}")
            raise HTTPException(status_code=400, detail=str(val_err))
        except Exception as exc:
            logger.error(f"[Artisera] Inference pipeline failure on '{filename}': {str(exc)}", exc_info=True)
            raise HTTPException(status_code=500, detail="An error occurred while processing the artisan image.")
        finally:
            await upload_file.close()

    return {
        "success": True,
        "count": len(results),
        "images": results,
    }


# ==============================================================================
# 9. HTML INTERFACE SERVING
# ==============================================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_index():
    """Serves the standalone Artisera AI Studio web UI for testing."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<h1>Artisera AI Studio</h1><p>index.html not found in service directory.</p>",
            status_code=404,
        )
    return FileResponse(index_path, media_type="text/html")


# ==============================================================================
# 10. DIRECT EXECUTION HELPER
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    # Listens on 0.0.0.0:8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
