from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.metrics import r2_score
import os

app = Flask(__name__)

# Make sure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load the trained models and their R² scores
try:
    with open('models/slr_model.pkl', 'rb') as f:
        slr_model = pickle.load(f)

    with open('models/mlr_model.pkl', 'rb') as f:
        mlr_model = pickle.load(f)
    
    # R² scores from training
    SLR_R2_SCORE = 0.9207
    MLR_R2_SCORE = 0.9842
    
    models_loaded = True
except (FileNotFoundError, EOFError) as e:
    print(f"Error loading models: {e}")
    models_loaded = False
    slr_model = None
    mlr_model = None
    SLR_R2_SCORE = 0
    MLR_R2_SCORE = 0

@app.route('/')
def home():
    return render_template('index.html', slr_r2=SLR_R2_SCORE, mlr_r2=MLR_R2_SCORE, models_loaded=models_loaded)

@app.route('/slr', methods=['GET', 'POST'])
def slr():
    prediction = None
    if not models_loaded:
        return render_template('slr.html', error="Models are not loaded. Please train the models first.", r2_score=SLR_R2_SCORE, models_loaded=models_loaded)
    
    if request.method == 'POST':
        try:
            engine_size = float(request.form['engine_size'])
            prediction = slr_model.predict([[engine_size]])[0]
        except:
            prediction = "Error: Please enter valid values"
    return render_template('slr.html', prediction=prediction, r2_score=SLR_R2_SCORE, models_loaded=models_loaded)

@app.route('/mlr', methods=['GET', 'POST'])
def mlr():
    prediction = None
    if not models_loaded:
        return render_template('mlr.html', error="Models are not loaded. Please train the models first.", r2_score=MLR_R2_SCORE, models_loaded=models_loaded)
    
    if request.method == 'POST':
        try:
            engine_size = float(request.form['engine_size'])
            cylinders = float(request.form['cylinders'])
            fuel_type = float(request.form['fuel_type'])
            prediction = mlr_model.predict([[engine_size, cylinders, fuel_type]])[0]
        except:
            prediction = "Error: Please enter valid values"
    return render_template('mlr.html', prediction=prediction, r2_score=MLR_R2_SCORE, models_loaded=models_loaded)

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app, binding to all interfaces (0.0.0.0)
    app.run(host='0.0.0.0', port=port, debug=False) 