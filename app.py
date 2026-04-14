import os
from io import BytesIO

import flask
from flask import render_template

from PIL import Image

from torch import argmax, load
from torch.cuda import is_available
from torch.nn import Sequential, Linear, SELU, Dropout
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torchvision.models import resnet50

# ------------------ CONFIG ------------------

UPLOAD_FOLDER = os.path.join('static', 'photos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

LABELS = ['None', 'Meningioma', 'Glioma', 'Pituitary']

device = "cuda" if is_available() else "cpu"

# ------------------ MODEL ------------------

resnet_model = resnet50(pretrained=True)

# Modify classifier
n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(
    Linear(n_inputs, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 4)   # NO activation here
)

resnet_model.to(device)

# IMPORTANT: fix path if needed
resnet_model.load_state_dict(load('Model/brain_tumor_model.pt', map_location=device))
resnet_model.eval()

# ------------------ PREPROCESS ------------------

transform = Compose([
    Resize((224, 224)),  # match training
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
    tensor = preprocess_image(image_bytes)
    tensor = tensor.to(device)

    y_hat = resnet_model(tensor)
    class_id = argmax(y_hat, dim=1)

    return str(int(class_id)), LABELS[int(class_id)]

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
        class_id, class_name = get_prediction(img_bytes)

        return render_template('pred.html', result=class_name)

# ------------------ ERROR ------------------

@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500

# ------------------ RUN ------------------

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)