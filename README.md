# 🌽 Maize Disease Detector (Hindi + English)

Upload a maize leaf photo and get the disease name, confidence,
Grad-CAM heatmap and treatment advice in Hindi.

## Classes
Blight, Common Rust, Gray Leaf Spot, Healthy

## Model
MobileNetV3-Large (transfer learning + fine-tuning), PyTorch.
Class-weighted loss and weighted sampling for the smaller Gray Leaf Spot class.

## Results (test set, 629 images)
- Accuracy: 96.7%
- Gray Leaf Spot recall: 87.2%
- Main confusion: Blight vs Gray Leaf Spot

## Run locally
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py

## Limitations
- Trained on a public dataset with clean images; real-field accuracy may be lower.
- Only knows 4 maize classes. Non-maize images can give wrong output.
- Treatment text is general information. Consult a KVK or agriculture officer.