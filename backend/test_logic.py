from core_ai import SkinCancerModel, predict_and_explain
import matplotlib.pyplot as plt

# 1. Init Model
model = SkinCancerModel(r"best_skin_model.pth")

# 2. Run Inference
with open(r"sample images/nv.jpeg", "rb") as f:
    result = predict_and_explain(model, f)

# 3. Show Result
print(f"Diagnosis: {result['class']}")
print(f"Confidence: {result['confidence']}%")
print(f"Uncertainty: {result['risk_flag']}")

plt.imshow(result['heatmap_image'])
plt.show()