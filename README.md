🚦 Automated Traffic Violation Monitoring with AI
Using Explainable Machine Learning (XAI) for Transparent Violation Detection

This project presents an AI-powered, explainable, and hybrid system for detecting common traffic violations such as overspeeding and red-light crossing. The system integrates machine learning, computer vision, and explainable AI techniques (SHAP & Grad-CAM) to ensure both high accuracy and transparent decision-making.

📌 Table of Contents

Overview

Features

Tech Stack

Dataset

Methodology

Explainable AI (XAI)

Model Performance

Results

How to Run

Applications

Future Work

Contributors

License

🧠 Overview

Urban traffic violations—especially speeding and red-light jumps—lead to congestion and accidents. Traditional monitoring is labor-intensive and lacks real-time detection and transparency.

This project introduces a hybrid AI system that:

Detects violations using machine learning on tabular features.

Uses computer vision (CNN + Grad-CAM) to analyze vehicle images.

Integrates Explainable AI (XAI) for feature-level and image-level transparency.

The goal is to build an accurate, scalable, and interpretable traffic monitoring system.

⭐ Key Features

✔️ Detect overspeeding and red-light violations
✔️ Machine Learning models: Random Forest, XGBoost, CatBoost, LightGBM, DNN
✔️ Image-based explainability using Grad-CAM
✔️ Tabular feature explanations using SHAP
✔️ Synthetic dataset generation for traffic scenarios
✔️ High model accuracy (90–95%)
✔️ Hybrid multimodal design (images + tabular data)

🛠️ Tech Stack
Component	Technology
ML Models	Random Forest, XGBoost, CatBoost
Deep Learning	ResNet18 (Grad-CAM)
Explainability	SHAP, Grad-CAM
Data Handling	Pandas, Numpy
Visualization	Matplotlib, Seaborn
Development	Python 3.x, Jupyter Notebook
📊 Dataset

Vehicle Images: Open Images Dataset (Cars subset)

Traffic Features (Synthetic):

Vehicle speed (km/h)

Distance to stop line (m)

Traffic light state (red/green)

Violation labels are generated using rule-based logic:

Speed > 80 km/h → Overspeeding

Distance < 2m + Red light → Red-light violation

🧬 Methodology
✔ Data Preprocessing

Standardization of numeric features

One-hot encoding for categorical features

Outlier handling (IQR-based winsorization)

SMOTE used for class imbalance

✔ Machine Learning Models Tested

Random Forest

XGBoost

LightGBM

CatBoost

Logistic Regression / SVM / KNN

Deep Neural Network

✔ Evaluation Metrics

Accuracy

Precision, Recall, F1-score

AUC-ROC

Confusion Matrix

🔍 Explainable AI (XAI)
1️⃣ SHAP (Tabular Explanations)

Identifies feature contribution for each prediction

Top influential features:

Vehicle Speed

Distance to stop line

Traffic light state

2️⃣ Grad-CAM (Image Explanations)

Highlights critical regions in vehicle images

Shows why the CNN predicted a violation

This combination increases transparency, trust, and legal accountability.

📈 Model Performance (Summary)
Model	Accuracy	Precision	Recall	F1-Score
XGBoost	0.93	0.95	0.98	0.96
CatBoost	0.91	0.93	0.98	0.95
Random Forest	0.92	0.91	0.90	0.95
DNN	0.91	—	—	—

XGBoost & CatBoost delivered the best overall performance.

📌 Results

Achieved 90–95% accuracy in violation prediction

SHAP revealed that speeding and stop-line distance are dominant predictors

Grad-CAM highlighted vehicle fronts, wheels, and road area near stop lines

Provided end-to-end transparency for each prediction

▶️ How to Run
1. Clone the Repository
git clone https://github.com/your-username/traffic-violation-ai.git
cd traffic-violation-ai

2. Install Dependencies
pip install -r requirements.txt

3. Run Jupyter Notebook
jupyter notebook

4. Execute Model Training

Open the file:
traffic_violation_detection.ipynb

🚀 Applications

Smart City Traffic Monitoring

Police Violation Analysis Tools

Automated Challan Generation

Red-light/speed camera integration

AI-based driver safety systems

🔮 Future Work

Integrate real-world video feed

Improve dataset using real traffic footage

Add more violation types (wrong-way, no-helmet, lane cutting)

Deploy as a real-time edge-camera system

Build a full web dashboard for live monitoring

👥 Contributors

Renatla Deekshith (2303A52104)

Nitesh Kumar (2303A52098)

📄 License

This project is open-source and available under the MIT License.
