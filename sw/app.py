from flask import Flask, render_template, request, jsonify

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from cnn_mnist import CNN

from PIL import Image
import re
import base64
import io

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def get_model():
    global model
    model = torch.load("./cnn_mnist.pt", weights_only=False)
    model.eval()
    #print(model)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    base64str = request.get_json(force=True)['base64str']
    #print(base64str)

    imgstr = re.search(r'base64,(.*)', str(base64str)).group(1)
    file = io.BytesIO(base64.b64decode(imgstr));
    img = Image.open(file).convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 1, 28, 28)

    data = Variable(torch.tensor((im2arr / 255), dtype = torch.float32))
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    pred_str = str(', '.join(map(str, pred.flatten().tolist())))
    #print(pred_str)

    response = {
        'prediction': pred_str
    }
    return jsonify(response)

print(" * Loading model...")
get_model()
print(" * Loading model done")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
