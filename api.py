from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image
import os
from typing import List
from search_engine import MultiModalSearch

# Initialize FastAPI app
app = FastAPI(title="Multi-Modal Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize search engine
search_engine = MultiModalSearch()

@app.post("/index/images")
async def index_images(images: List[UploadFile] = File(...)):
    try:
        image_paths = []
        for image in images:
            # Read and validate image
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            
            # Save image
            path = os.path.join(UPLOAD_DIR, image.filename)
            img.save(path)
            image_paths.append(path)
        
        # Index images
        search_engine.index_images(image_paths)
        return JSONResponse(
            content={"message": f"Indexed {len(image_paths)} images successfully"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/text")
async def search_by_text(query: str, k: int = 5):
    try:
        results = search_engine.search_by_text(query, k)
        return JSONResponse(
            content={"results": results},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}