
# 🧠 Customer Segmentation using K-Means (2D & 3D)

This project demonstrates **unsupervised machine learning** using **K-Means clustering** for customer segmentation, followed by **real-time prediction and visualization using Streamlit**.

It is designed for:

* Classroom learning
* Hands-on ML practice
* Deployment understanding

---

## 📌 Project Overview

We build two clustering models:

| Model          | Features                           |
| -------------- | ---------------------------------- |
| **2D K-Means** | Annual Income, Spending Score      |
| **3D K-Means** | Age, Annual Income, Spending Score |

The trained models are then used inside a **Streamlit web application** to:

* Predict customer segments in real time
* Visualize clusters in 2D and 3D
* Explain clusters to non-technical users
* Perform batch predictions using CSV files

---

## 📁 Project Structure

```
customer-segmentation/
│
├── app.py                  # Streamlit application
├── kmeans_2d.pkl           # Trained 2D K-Means model
├── scaler_2d.pkl           # Scaler for 2D model
├── kmeans_3d.pkl           # Trained 3D K-Means model
├── scaler_3d.pkl           # Scaler for 3D model
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset Used

**Mall Customer Segmentation Dataset**

Features:

* Age
* Annual Income (k$)
* Spending Score (1–100)

This dataset is commonly used for teaching clustering and customer analytics.

---

## ⚙️ Model Training (Colab / Notebook)

### Steps Performed:

1. Load and explore the dataset
2. Feature selection (2D and 3D)
3. Data scaling using `StandardScaler`
4. Optimal cluster selection using:

   * Elbow Method
   * Silhouette Score
5. Train K-Means models
6. Save models using `joblib`

Saved files:

```python
joblib.dump(kmeans, "kmeans_model.pkl") ##2D Kmean_model
joblib.dump(scaler, "scaler.pkl") 

joblib.dump(kmeans_3d, "kmeans_3d_model.pkl") ## 3D Kmean_model
joblib.dump(scaler_3d, "scaler_3d.pkl")
```

---

## 🚀 Running the Streamlit App

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run the App

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Streamlit App Features

### 🔹 Model Selection

* Choose **2D** or **3D** clustering model

### 🔹 Real-Time Prediction

* Input Age, Income, Spending Score
* Instantly predict customer segment

### 🔹 Cluster Visualization

* 2D scatter plot
* 3D interactive plot

### 🔹 Customer Personas

Clusters are explained in simple language:

* High income, high spending
* Cautious customers
* Budget-conscious users
* Premium buyers

### 🔹 Batch Prediction

* Upload a CSV file
* Automatically assign customer segments
* Download prediction results

---

## 📂 CSV Upload Format

### 2D Model

```csv
Annual Income (k$),Spending Score (1-100)
```

### 3D Model

```csv
Age,Annual Income (k$),Spending Score (1-100)
```

---

## 🎓 Learning Outcomes

Students will learn:

* Unsupervised learning concepts
* K-Means clustering
* Feature scaling importance
* Model saving & loading
* Real-time ML inference
* Deployment using Streamlit
* Interpreting clusters for business users

---

## 🛠️ Technologies Used

* Python
* Scikit-Learn
* Pandas & NumPy
* Plotly
* Streamlit
* Joblib

---

## 📌 Future Improvements

* Cluster confidence scores
* Model comparison dashboard
* Streamlit Cloud deployment
* Dynamic cluster selection

---

## 📄 License

This project is for **educational purposes**.

---

## 👩‍🏫 Instructor Note

This project is intentionally designed to:

* Be beginner-friendly
* Encourage problem-solving
* Bridge ML theory with real-world deployment



---

# 🚀 Deployment on Streamlit Cloud

**Customer Segmentation (2D & 3D K-Means)**

---

## 1️⃣ Prerequisites

Before deployment, ensure you have:

* A **GitHub account**
* A **public GitHub repository**
* All required project files committed
* A **Streamlit Cloud account**
  👉 [https://streamlit.io/cloud](https://streamlit.io/cloud)

---

## 2️⃣ Final Project Structure (REQUIRED)

Your GitHub repository **must** look like this:

```
customer-segmentation/
│
├── app.py
├── kmeans_2d.pkl
├── scaler_2d.pkl
├── kmeans_3d.pkl
├── scaler_3d.pkl
├── requirements.txt
└── README.md
```

⚠️ **Important**

* Filenames must match **exactly**
* Do NOT place files inside subfolders unless you update paths in `app.py`

---

## 3️⃣ `requirements.txt` (Final Check)

```txt
streamlit
numpy
pandas
scikit-learn
joblib
plotly
```

No extra packages needed.

---

## 4️⃣ Push Project to GitHub

### Step 1: Initialize Git (if not done)

```bash
git init
git add .
git commit -m "Initial commit: K-Means clustering Streamlit app"
```

---

### Step 2: Create GitHub Repository

* Go to **GitHub → New Repository**
* Name it: `customer-segmentation`
* Keep it **Public**
* Do NOT add README (already exists)

---

### Step 3: Push Code

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/customer-segmentation.git
git push -u origin main
```

---

## 5️⃣ Deploy on Streamlit Cloud

### Step 1: Open Streamlit Cloud

👉 [https://share.streamlit.io/](https://share.streamlit.io/)

---

### Step 2: Click **“New app”**

Fill the form as follows:

| Field          | Value                                 |
| -------------- | ------------------------------------- |
| Repository     | `your-username/customer-segmentation` |
| Branch         | `main`                                |
| Main file path | `app.py`                              |
| Python version | Auto                                  |

Click **Deploy**.

---

## 6️⃣ First-Time Deployment Notes

### ⏳ Initial Build

* Takes **1–2 minutes**
* Streamlit installs dependencies automatically
* Models (`.pkl`) are loaded at runtime

### ✅ Successful Deployment

You’ll receive a **public URL**, e.g.:

```
https://customer-segmentation.streamlit.app
```

---

## 7️⃣ Common Deployment Issues & Fixes

### ❌ Error: `FileNotFoundError: kmeans_2d.pkl`

✔️ Fix:

* Ensure `.pkl` files are **committed to GitHub**
* Filenames match exactly
* They are in the **root directory**

---

### ❌ App crashes on start

✔️ Fix:

* Check **Logs** in Streamlit Cloud
* Confirm all libraries are in `requirements.txt`
* Remove unnecessary imports

---

### ❌ CSV Upload Not Working

✔️ Fix:

* Ensure column names match:

  ```
  Age
  Annual Income (k$)
  Spending Score (1-100)
  ```

---

## 8️⃣ Performance & Optimization Tips (Optional Teaching Point)

For smoother performance:

* Limit grid size in 3D plots
* Cache models (optional)

```python
@st.cache_resource
def load_model(path):
    return joblib.load(path)
```

---

## 9️⃣ Classroom Deployment Workflow (Recommended)

| Phase | Action                     |
| ----- | -------------------------- |
| Lab 1 | Train K-Means in Colab     |
| Lab 2 | Save models                |
| Lab 3 | Build Streamlit app        |
| Lab 4 | Deploy on Streamlit Cloud  |
| Lab 5 | Student demo & explanation |

---

## 10️⃣ How Students Can Present This Project

Students should explain:

1. Why clustering is unsupervised
2. Why scaling is required
3. Difference between 2D and 3D models
4. How clusters map to customer personas
5. How ML models are deployed

---

## ✅ Final Checklist (Before Submission)

✔️ App runs locally
✔️ All `.pkl` files present
✔️ GitHub repo is public
✔️ `requirements.txt` correct
✔️ Streamlit app deployed successfully



