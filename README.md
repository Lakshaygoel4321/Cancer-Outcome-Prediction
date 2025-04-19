# **Cancer Outcome Prediction** 🧬⚕️  

This repository presents a complete **MLOps pipeline** designed to predict cancer treatment outcomes using patient data. The project includes data ingestion, transformation, model training, and a full-stack deployment using **Flask (backend)** and **React (frontend)** — all deployed without CI/CD or MongoDB integration.

---

## 🛠 **Tools & Technologies**  
This project utilizes:  
- **Python**: Core language for backend and ML pipeline  
- **Pandas & NumPy**: Data handling and numerical operations  
- **Scikit-learn**: Machine learning model development  
- **Flask**: API backend to serve the trained model  
- **React**: Frontend UI for entering patient information and viewing predictions  
- **Pickle & Joblib**: Model serialization  
- **Render**: Deployment platform for both frontend and backend  

---

## 📊 **Dataset**  
The dataset contains anonymized patient information including demographics, cancer details, and treatment type. It is used to predict the outcome as:  
- `0`: Deceased  
- `1`: Recovered  
- `2`: Under Treatment  

---

## ⚙️ **Project Workflow**

This MLOps pipeline consists of the following modular steps:  

1️⃣ **Data Ingestion**  
- Implemented in: `components/data_ingestion.py`  
- Loads raw CSV data for further processing.

2️⃣ **Data Transformation**  
- Implemented in: `components/data_transformation.py`  
- Cleans data, encodes categorical variables, scales numerical features.

3️⃣ **Model Training**  
- Implemented in: `components/model_trainer.py`  
- Trains a classification model using scikit-learn and saves it as `model.pkl`.

4️⃣ **Pipeline Orchestration**  
- Main pipeline logic in: `pipeline/training_pipeline.py`  
- Ties together ingestion, transformation, and training steps.

5️⃣ **Web Application**  
- **Backend**: `app.py` (Flask) loads the model and preprocessor, exposes `/predict` API.  
- **Frontend**: React app in `/cancer-predictor/`, sends user input to backend and shows the result.

---

## 🖥 **How to Run the Project Locally**

### Step 1: Clone the Repository  
```bash
git clone https://github.com/yourusername/cancer-outcome-prediction.git
cd cancer-outcome-prediction
```

### Step 2: Create and Activate Virtual Environment  
```bash
python -m venv cancer_env
cancer_env\Scripts\activate  # Windows
# or
source cancer_env/bin/activate  # macOS/Linux
```

### Step 3: Install Backend Dependencies  
```bash
pip install -r requirements.txt
```

### Step 4: Run the Training Pipeline  
```bash
python pipeline/training_pipeline.py
```

### Step 5: Run the Flask Backend  
```bash
python app.py
```
Flask runs on: **http://localhost:5000**

### Step 6: Run the React Frontend  
```bash
cd cancer-predictor
npm install
npm start
```
React runs on: **http://localhost:3000**

---

## 📁 **Project Structure**

```
cancer-outcome-prediction/
├── app.py
├── requirements.txt
├── model.pkl
├── preprocessor.pkl
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   └── model_trainer.py
├── pipeline/
│   └── training_pipeline.py
├── cancer-predictor/      # React frontend
│   ├── package.json
│   ├── src/
│   └── public/
```

---

## 🌍 **Deployment**

The application is deployed on [Render](https://render.com):  

- 🔗 **Frontend** (React): https://mlops-project-11.onrender.com/

---

## 💡 **Key Features**

✔ Modular MLOps pipeline with reusability  
✔ End-to-end deployment (manual, without CI/CD)  
✔ Clean and responsive React UI  
✔ Real-time predictions served via Flask API  
✔ Fully open-source and customizable

---

## 📂 **Git Commands for Version Control**  
```bash
git add .
git commit -m "Updated cancer predictor pipeline"
git push origin main
```

---

## 🤝 **Contributions**

Contributions are welcome!  
Feel free to fork this repo, submit pull requests, or open issues to collaborate on improvements.

---

## 📄 **License**  
This project is licensed under the **MIT License**.
