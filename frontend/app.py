from flask import Flask, request, jsonify, render_template
import os
from ultralytics import YOLO
from PIL import Image
import torch
import base64
from io import BytesIO

app = Flask(__name__)
app.static_folder = 'static'
model = YOLO('D:/Python/大作业/人工智能/runs/detect/train3/weights/best.pt')


@app.route('/')
def index():
    return render_template('index.html')


def pred(image_path):
    results = model.predict(image_path, conf=0.5, device='cuda:0', classes=[46, 47, 49])

    for i, r in enumerate(results):
        im_bgr = r.plot(conf=False)
        im_rgb = Image.fromarray(im_bgr[:, :, ::-1])
        buffered = BytesIO()
        im_rgb.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file:
        filepath = f'D:/Python/大作业/人工智能/AI大作业/backend/pred/{file.filename}'
        print(filepath)
        file.save(filepath)

        result_image_base64 = pred(filepath)

        return jsonify({'success': True, 'image_base64': result_image_base64})


if __name__ == '__main__':
    app.run(debug=False, port=8000)
