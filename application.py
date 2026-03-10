from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.mlproject.components import data_transformation
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.utils import read_sql_data
from src.mlproject.components.data_transformation import DataTransformationConfig
from src.mlproject.utils import save_object
import sys
from src.mlproject.utils import evaluate_models
from src.mlproject.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.mlproject.pipelines.prediction_pipeline import PredictPipeline, CustomData




application = Flask(__name__)
app = application

@app.route('/predictions', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=request.form.get("reading_score"),
                writing_score=request.form.get("writing_score")
            )

            final_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(final_data)

            return render_template('home.html', prediction=prediction[0])

        except Exception as e:
            return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')