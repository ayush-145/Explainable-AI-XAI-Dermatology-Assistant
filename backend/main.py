from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
from PIL import Image
import numpy as np

# Import our custom AI logic from the other file
try:
    from core_ai import SkinCancerModel, predict_and_explain
except ImportError:
    print("="*50)
    print("ERROR: core_ai.py not found.")
    print("Please make sure 'core_ai.py' is in the same directory as 'main.py'.")
    print("="*50)
    exit()

# 1. Initialize FastAPI App
app = FastAPI(
    title="DermaNet-X Diagnostic API",
    description="An Explainable AI (XAI) backend for Skin Lesion Classification.",
    version="1.0.0"
)

# 2. Add CORS Middleware
# This allows our React Frontend (on a different port) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    # In production, change "*" to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Load the AI Model
# This is a best practice: load the model once on startup, not on every request.
MODEL_PATH = "best_skin_model.pth"
try:
    # We use CPU for inference as this is a web server.
    # We can't rely on the server having a GPU.
    ai_model = SkinCancerModel(MODEL_PATH, device="cpu") 
    print(f"✅ Model '{MODEL_PATH}' loaded successfully on CPU.")
except FileNotFoundError:
    print(f"❌ FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please download 'best_skin_model.pth' from Kaggle and place it in the 'backend' folder.")
    ai_model = None
except Exception as e:
    print(f"❌ FATAL ERROR: An unexpected error occurred while loading the model: {e}")
    ai_model = None

# --- Helper Function ---
def image_to_base64(image_array):
    """
    Converts a Numpy/CV2/PIL image (from our core_ai) 
    to a Base64 string that can be sent in a JSON response.
    """
    try:
        if isinstance(image_array, np.ndarray):
            # If it's a numpy array (from cv2), convert to PIL
            image_pil = Image.fromarray(image_array)
        else:
            # Assume it's already a PIL Image
            image_pil = image_array
            
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        # Return the string in a format that <img> tags can read
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to Base64: {e}")
        return None

# 4. Define API Endpoints
@app.get("/")
def home():
    """A simple health-check endpoint."""
    if ai_model is None:
        return {"status": "error", "message": "Model not loaded. API is not operational."}
    return {"status": "ok", "message": "DermaNet-X API is running. Model is loaded."}


@app.post("/predict")
async def handle_prediction(file: UploadFile = File(...)):
    """
    The main endpoint.
    1. Receives an image.
    2. Runs prediction and Grad-CAM explanation.
    3. Returns JSON with diagnosis, confidence, and Base64 heatmap.
    """
    if ai_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process requests.")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG.")

    try:
        # Read the image file as bytes
        image_bytes = await file.read()
        
        # --- This is where all the magic happens ---
        # We pass the raw bytes to our logic function
        result = predict_and_explain(ai_model, io.BytesIO(image_bytes))
        
        # Convert the numpy array heatmap to a Base64 string
        heatmap_b64 = image_to_base64(result["heatmap_image"])
        
        if heatmap_b64 is None:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap image.")

        # Return the final JSON response
        return JSONResponse(content={
            "filename": file.filename,
            "diagnosis": result["class"],
            "confidence": result["confidence"],
            "uncertainty_score": result["uncertainty_score"],
            "risk_flag": result["risk_flag"],
            "heatmap_base64": heatmap_b64
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# 5. Run the App
if __name__ == "__main__":
    if ai_model is None:
        print("\n--- WARNING: API IS STARTING WITHOUT A MODEL. '/predict' WILL NOT WORK. ---")
    
    # This makes the server run. '--reload' is great for development.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)