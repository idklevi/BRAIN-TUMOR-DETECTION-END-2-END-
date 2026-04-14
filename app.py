import os
from io import BytesIO

import flask
from flask import render_template

import gdown

from PIL import Image

from torch import argmax, load
from torch.cuda import is_available
from torch.nn import Linear
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import resnet50

# ------------------ CONFIG ------------------

UPLOAD_FOLDER = os.path.join('static', 'photos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "models/bt_resnet50_model.pt"
os.makedirs("models", exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1K0nFmRKfQGJGrylwrKIi_1fGZ4PMTzzF"
    gdown.download(url, MODEL_PATH, quiet=False)

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ✅ 3 classes only
LABELS = ['Meningioma', 'Glioma', 'Pituitary']

device = "cuda" if is_available() else "cpu"

# ------------------ MODEL ------------------

resnet_model = resnet50(pretrained=True)

# ✅ MUST MATCH TRAINING EXACTLY
resnet_model.fc = Linear(resnet_model.fc.in_features, 3)

resnet_model.to(device)

resnet_model.load_state_dict(load(MODEL_PATH, map_location=device))
resnet_model.eval()

# ------------------ PREPROCESS ------------------

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = preprocess_image(image_bytes).to(device)

    y_hat = resnet_model(tensor)
    class_id = argmax(y_hat, dim=1)

    return LABELS[int(class_id)]

# ------------------ ROUTES ------------------

@app.route('/', methods=['GET'])
def main():
    return render_template('DiseaseDet.html')

@app.route("/uimg", methods=['GET', 'POST'])
def uimg():
    if flask.request.method == 'GET':
        return render_template('uimg.html')

    if flask.request.method == 'POST':
        file = flask.request.files.get('file')

        if not file or not allowed_file(file.filename):
            return "Invalid file", 400

        img_bytes = file.read()
        class_name = get_prediction(img_bytes)

        return render_template('pred.html', result=class_name)

# ------------------ ERROR ------------------

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500

# ------------------ RUN ------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)