<div align="center">

# 🧠 Brain Tumor Detection — End to End

> A deep learning web application that classifies brain tumors from MRI scans into three types using a fine-tuned ResNet50 model.
> Covers the complete pipeline — from raw dataset preprocessing to a deployed Flask app.

</div>

---

## ⚠️ Medical Disclaimer

This project is for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**. Always consult a qualified medical professional.

---

## 🔬 About the Project

Brain tumors are among the most critical medical conditions, where accurate classification plays a major role in treatment decisions.

This project explores how **transfer learning with ResNet50** can be used to classify MRI scans into tumor categories with high accuracy.

Unlike simple model demos, this project is **end-to-end**, including:

* Processing raw `.mat` MRI data
* Converting and organizing dataset
* Training and fine-tuning a deep learning model
* Building a web interface for real-time predictions
* Preparing the system for deployment

---

## ⚙️ How It Works

```
User Uploads MRI Image
        ↓
Preprocessing (Resize → Normalize → Tensor)
        ↓
ResNet50 Model Inference
        ↓
3-Class Output (Softmax)
        ↓
Predicted Tumor Type Displayed
```

### Classes:

* **Glioma**
* **Meningioma**
* **Pituitary**

---

## 📊 Dataset

| Property         | Details                                |
| ---------------- | -------------------------------------- |
| **Name**         | Brain Tumor Dataset                    |
| **Author**       | Jun Cheng                              |
| **Source**       | Figshare                               |
| **Total Images** | 3,064                                  |
| **Patients**     | 233                                    |
| **Format**       | `.mat` (MATLAB) → converted to `.jpg`  |
| **Task**         | Multi-class classification (3 classes) |

---

### Class Distribution

| Class          | Description                               | Samples |
| -------------- | ----------------------------------------- | :-----: |
| **Glioma**     | Aggressive tumor from glial cells         |   1426  |
| **Meningioma** | Tumor from brain membranes (often benign) |   708   |
| **Pituitary**  | Tumor in pituitary gland                  |   930   |

---

### `.mat` File Structure

Each `.mat` file contains:

| Field                | Description                |
| -------------------- | -------------------------- |
| `cjdata.image`       | MRI image matrix           |
| `cjdata.label`       | Tumor label (1, 2, 3)      |
| `cjdata.tumorBorder` | Tumor boundary coordinates |
| `cjdata.tumorMask`   | Binary tumor mask          |

---

### Data Preprocessing

* Converted `.mat` → `.jpg`
* Organized into class folders
* Applied transformations:

  * Resize to 224×224
  * Normalization (ImageNet mean/std)

---

## 🏗️ Model Architecture

Instead of training from scratch, this project uses **transfer learning**.

### Base Model:

* **ResNet50 (ImageNet pretrained)**

### Modifications:

* Final fully connected layer replaced for **3-class output**

```
Input (224×224×3)
      ↓
ResNet50 Backbone
      ↓
Fully Connected Layer (2048 → 3)
      ↓
Output: Glioma / Meningioma / Pituitary
```

---

## 📈 Model Performance

| Metric            | Value |
| ----------------- | :---: |
| **Accuracy**      |  ~99% |
| **Glioma F1**     |  ~99% |
| **Meningioma F1** |  ~98% |
| **Pituitary F1**  |  ~99% |

⚠️ Notes:

* High accuracy is dataset-specific
* Model may not generalize to real-world clinical data
* Class imbalance slightly affects meningioma performance

---

## 🖥️ Web Application

A Flask-based web interface enables real-time predictions.

### Features:

* Upload MRI images
* Instant classification
* Clean UI (responsive for desktop & mobile)
* End-to-end inference pipeline

---

## 🧠 Model Handling

The trained model (~90MB) is **not included in the repository**.

### Instead:

* Stored on Google Drive
* Downloaded dynamically at runtime using `gdown`

```python
gdown.download(url, MODEL_PATH)
```

This keeps the repository lightweight and deployment-friendly.

---

## 📁 Project Structure

```
project/
│
├── models/                 # downloaded model
├── static/                 # css, uploads
├── templates/              # html pages
├── app.py                  # flask app
├── Dockerfile              # deployment
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/idklevi/BRAIN-TUMOR-DETECTION-END-2-END.git
cd "BRAIN TUMOR DETECTION [END 2 END]"

python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

### Key Points:

* Uses `gunicorn` for production
* Model downloaded at startup
* No large files in repository
* Lightweight frontend

---

## 🛠️ Tech Stack

| Layer            | Technology           |
| ---------------- | -------------------- |
| Language         | Python               |
| Deep Learning    | PyTorch, Torchvision |
| Model            | ResNet50             |
| Backend          | Flask                |
| Frontend         | HTML, CSS, Jinja2    |
| Image Processing | PIL                  |
| Deployment       | Docker, Render       |

---

## ⚠️ Limitations

* Small dataset size
* Class imbalance
* No clinical validation
* Not suitable for real medical diagnosis

---

## 📌 Future Improvements

* Add confidence score display
* Add Grad-CAM visualization
* Improve dataset size
* Enhance UI/UX
* Optimize inference speed

---

## 📚 References

* Jun Cheng Brain Tumor Dataset (Figshare)
* ResNet Paper (He et al., 2015)
* PyTorch Transfer Learning Docs

---

<div align="center">

Built as a complete deep learning pipeline 🚀

</div>
