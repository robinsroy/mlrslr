# CO₂ Emission Prediction Web App

A Flask web application for predicting CO₂ emissions of vehicles using:
- Simple Linear Regression (SLR) based on Engine Size
- Multiple Linear Regression (MLR) based on Engine Size, Cylinders, and Fuel Type

## Features

- Two separate prediction interfaces (SLR and MLR)
- Interactive UI with form validation
- Display of model accuracy
- Responsive design

## Project Structure

```
├── app.py              # Main Flask application
├── wsgi.py             # WSGI entry point for production
├── train_models.py     # Script for training and saving models
├── requirements.txt    # Project dependencies
├── models/             # Saved ML models
│   ├── slr_model.pkl
│   └── mlr_model.pkl
├── data/               # Dataset
│   └── co2_emission.csv
└── templates/          # HTML templates
    ├── base.html       # Base template with layout
    ├── index.html      # Homepage
    ├── slr.html        # SLR prediction page
    └── mlr.html        # MLR prediction page
```

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train models:
   ```
   python train_models.py
   ```

3. Run the application:
   ```
   python app.py
   ```

## Deployment

This application is configured for deployment on Render. 