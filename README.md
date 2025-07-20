# multiclass-fish-classification

🐠 Multiclass Fish Image Classifier
A deep learning-based image classification project to identify various fish species using custom-trained CNN and transfer learning models. Deployed with an interactive Streamlit web application.

📌 Project Overview
This project focuses on classifying fish images into multiple species using deep learning. We trained a CNN from scratch and fine-tuned 5 pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0) to compare performance and accuracy.

The best-performing model is deployed as a user-friendly web app using Streamlit.

🎯 Problem Statement
Classify fish images into the correct species.

Improve accuracy using transfer learning.

Build an interactive web app for real-time image prediction.

Compare performance across different model architectures.

🧠 Skills You’ll Learn
Deep Learning (CNNs & Transfer Learning)

TensorFlow / Keras

Image Preprocessing & Augmentation

Model Evaluation (Precision, Recall, F1-score)

Streamlit Deployment

Visualization (Accuracy/Loss, Confusion Matrix)

📁 Dataset
Images are categorized in subfolders, each representing a fish species.

Preprocessed using ImageDataGenerator with:

Rescaling

Zoom, Flip, Rotation augmentations

🏗️ Project Structure
bash
Copy
Edit
.
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── mobilenet_model.h5
│   ├── resnet_model.h5
│   └── ...
├── class_indices.json
├── app.py                 # Streamlit app
├── training_script.py     # Model training
├── evaluation_script.py   # Metrics + plots
├── README.md
└── requirements.txt
🚀 Model Architectures Used
✅ CNN (From Scratch)

✅ VGG16

✅ ResNet50

✅ MobileNet (Deployed)

✅ InceptionV3

✅ EfficientNetB0

📊 Evaluation Metrics
Each model was evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

📈 Training/validation curves were also plotted for comparison.

🌐 Streamlit Web App
Features:
Upload a fish image

See predicted species with confidence score

View top-3 predictions

Beautiful ocean-themed UI

Run the app locally:
bash
Copy
Edit
streamlit run app.py
✅ Installation
bash
Copy
Edit
git clone https://github.com/yourusername/deep-fish-classifier.git
cd deep-fish-classifier
pip install -r requirements.txt
🛠️ Requirements
nginx
Copy
Edit
tensorflow
streamlit
numpy
pillow
matplotlib
scikit-learn
📸 Sample Prediction

📬 Future Improvements
Add species info cards (habitat, size, facts)

Allow model switching in Streamlit app

Deploy to HuggingFace Spaces or Streamlit Cloud

📚 License
This project is open-source under the MIT License.
