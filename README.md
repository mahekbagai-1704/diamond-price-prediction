# 💎 Diamond Price Prediction

An end-to-end Machine Learning project that predicts diamond prices using a **Random Forest Regressor** and an interactive **Streamlit web application**.

## 🌐 Live Demo

🚀 **https://huggingface.co/spaces/Mahek-Bagai/Diamond_Price_prediction_App**

## 📸 Application Preview

### 🏠 Dashboard

![Dashboard]( )

### 💎 Price Prediction

![Price Prediction]( )

### 📊 Dataset & Model Information

![Dataset & Model Information]( )

## 📌 Overview

This project uses diamond characteristics such as **carat, cut, color, clarity, depth, table, and dimensions** to predict diamond prices.

The project covers the complete Machine Learning workflow:

**Data → Preprocessing → Encoding → Model Training → Evaluation → Model Saving → Streamlit Deployment**

## 🤖 Model

**Random Forest Regressor**

Categorical features (`cut`, `color`, and `clarity`) are transformed using **Label Encoding**.

The trained model and encoders are stored in:

    diamond_model.pkl

The model predicts log-transformed prices, which are converted back to the original price scale for the final prediction.

## 📊 Features

- Carat
- Cut
- Color
- Clarity
- Depth
- Table
- X, Y, Z dimensions

## 🛠️ Tech Stack

**Python • Pandas • NumPy • Scikit-learn • Streamlit • Matplotlib • Seaborn • Jupyter Notebook**

## ▶️ Run Locally

    git clone https://github.com/mahekbagai-1704/diamond-price-prediction.git
    cd diamond-price-prediction
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    streamlit run Diamond_Price_Prediction_App.py

## 📂 Project Structure

    diamond-price-prediction/
    │
    ├── Diamond_Price_Prediction_App.py
    ├── diamond_model.pkl
    ├── diamonds.csv
    ├── processed_diamonds.csv
    ├── diamonds_project.ipynb
    ├── .gitignore
    └── README.md

---

⭐ Built as part of my Data Science and Machine Learning learning journey.
