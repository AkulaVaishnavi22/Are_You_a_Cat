# 🐱 Are You a Cat? (Cat, Dog, & Human Classification Hub)

An interactive computer vision web application and MLOps pipeline built with **Streamlit**, **TensorFlow/Keras**, and **MLflow**. The application classifies input images into three categories—**Cats**, **Dogs**, or **Humans**—while tracking model metrics, artifacts, and training runs using an MLflow tracking server backed by SQLite.

---

## 📌 Features

- **📸 Multi-Class Image Classification**: Classifies uploaded or local images across three classes (`cats`, `dogs`, `humans`).
- **🧠 Model Management**: Dynamically detects available pre-trained models (`.h5` files) in the `models/` directory.
- **📊 Integrated MLflow Tracking**: Configured with a local SQLite backend (`sqlite:///mlflow.db`) to record model experiments, performance metrics, and baseline runs (`Cat_Dog_Human_Classification_Tracker`).
- **🎨 Interactive UI**: Simple, clean Streamlit web dashboard for real-time predictions and model management.

---

## 📂 Repository Structure

```text
Are_You_a_Cat-main/
├── data/                    # Dataset directory
│   ├── cats/                # Cat images
│   ├── dogs/                # Dog images
│   └── humans/              # Human images
├── models/                  # Saved Keras model files (.h5)
│   ├── baseline_cat_model.h5
│   └── model_v_20260724_210709.h5
├── mlruns/                  # MLflow experiment artifacts & metadata
├── app.py                   # Main Streamlit web application
├── train_baseline.py        # Model training script
├── validate_data.py         # Data validation utility
├── clean_duplicates.py      # Dataset cleanup helper
├── requirements.txt         # Project dependencies
├── Dockerfile               # Docker container configuration
└── mlflow.db                # SQLite database for MLflow experiment logs
🛠️ Installation & Setup Guide
Follow these steps to set up and run the project on your local machine.

1️⃣ Prerequisites
Make sure you have Python 3.10+ installed. Check your version with:

Bash
python --version
2️⃣ Download / Clone the Repository
Extract the downloaded ZIP file or clone the repo using Git:

Bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/Are_You_a_Cat.git](https://github.com/YOUR_GITHUB_USERNAME/Are_You_a_Cat.git)
cd Are_You_a_Cat-main
3️⃣ Create a Virtual Environment
Windows (PowerShell):
PowerShell
python -m venv venv
.\venv\Scripts\activate
macOS / Linux:
Bash
python3 -m venv venv
source venv/bin/activate
4️⃣ Install Dependencies
Install all required libraries using requirements.txt:

Bash
pip install -r requirements.txt
🚀 How to Run the Application
Launch the Streamlit Web UI
Run the following command in your activated terminal:

Bash
streamlit run app.py
Once running, open your web browser and navigate to:
👉 http://localhost:8501

🧪 Training & Data Pipeline
Data Cleanup: Run clean_duplicates.py to strip out duplicate images in the dataset before training.

Data Validation: Execute validate_data.py to ensure dataset integrity.

Train Baseline Model: Run train_baseline.py to train a new classifier and automatically log run parameters and metrics directly to MLflow.

📊 Viewing MLflow Experiment Logs
To inspect experiment logs, loss curves, and parameter comparisons locally:

Bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
Then open http://localhost:5000 in your browser.

🐳 Running with Docker (Optional)
If you prefer running the application inside a container:

Bash
# Build the Docker image
docker build -t are-you-a-cat .

# Run the container
docker run -p 8501:8501 are-you-a-cat
