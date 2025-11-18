import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F

# ==========================================
# 1. MODEL LOADER
# ==========================================
class SkinCancerModel:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Preprocessing must match Kaggle training exactly
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, path):
        print(f"Loading model from {path}...")
        # Initialize empty EfficientNet
        model = models.efficientnet_b0(weights=None)
        # Recreate the classifier head
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 7)
        # Load weights
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        return model

    def preprocess(self, image_bytes):
        """Converts raw bytes to Tensor"""
        image = Image.open(image_bytes).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device), image

# ==========================================
# 2. GRAD-CAM EXPLAINABILITY
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook into the gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        # 1. Forward Pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # 2. Backward Pass (Zero out previous gradients)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 3. Get Gradients & Activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        # 4. Weight the channels
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # 5. Create Heatmap
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap) # Remove negative values (ReLU)
        heatmap /= torch.max(heatmap) # Normalize
        
        return heatmap.cpu().numpy()

    @staticmethod
    def overlay_heatmap(heatmap, original_image, alpha=0.4):
        """Overlays the heatmap on the original PIL image"""
        img_np = np.array(original_image)
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Convert to color map
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose
        overlay = cv2.addWeighted(heatmap, alpha, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ==========================================
# 3. UNCERTAINTY & PREDICTION UTILS
# ==========================================
def predict_and_explain(wrapper, image_bytes):
    """
    Main function to be called by API.
    Returns: Prediction, Confidence, Uncertainty Flag, Heatmap Image
    """
    # A. Prediction
    img_tensor, original_img = wrapper.preprocess(image_bytes)
    output = wrapper.model(img_tensor)
    probabilities = F.softmax(output, dim=1)
    
    top_p, top_class = probabilities.topk(1, dim=1)
    confidence = top_p.item()
    class_idx = top_class.item()
    
    # B. Uncertainty Quantification (Entropy)
    # Higher Entropy = More confusion
    entropy = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=1).item()
    uncertainty_flag = "High" if entropy > 1.1 else "Low" # 1.1 is an arbitrary threshold for 7 classes
    
    # C. Explainability (Grad-CAM)
    # EfficientNet features are in model.features. The last block is [-1]
    cam = GradCAM(wrapper.model, wrapper.model.features[-1])
    heatmap_raw = cam.generate_heatmap(img_tensor, class_idx)
    heatmap_img = cam.overlay_heatmap(heatmap_raw, original_img)
    
    # Class names mapping
    labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    
    return {
        "class": labels[class_idx],
        "confidence": round(confidence * 100, 2),
        "uncertainty_score": round(entropy, 4),
        "risk_flag": uncertainty_flag,
        "heatmap_image": heatmap_img # Numpy array (needs converting to bytes for API)
    }