# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
from src.states import CONTEXT
import os

app = Flask(__name__)
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app, path = None)
metrics.start_http_server(5099)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file1 = request.files['file1']    
    # Сохраняем загруженные файлы для использования в предсказании
    file1.save(os.path.join('static/images', file1.filename))
    prediction_result = CONTEXT.classifier.inference(
        os.path.join('static/images', file1.filename))
    os.remove(os.path.join('static/images', file1.filename))
    if prediction_result[1] == 0:
        return jsonify({'prediction': prediction_result[0]}), 200
    else:
        print(0)
        return ":(", 500

if __name__ == '__main__':
    if not os.path.exists('static/images'):
        os.makedirs('static/images')
    
    app.run("0.0.0.0", 5000) #debug=True
