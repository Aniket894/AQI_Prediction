# src/training_pipeline.py
import os
from src.components.data_ingestion import ingest_data
from src.components.data_transformation import transform_data
from src.components.model_training import train_models
from src.logger import logging

def run_pipeline():
    logging.info("Starting the training pipeline")

    try:
        # Step 1: Data Ingestion
        logging.info("Running data ingestion step")
        ingest_data("notebook/data/dailyDataAQI.csv")
        logging.info("Data ingestion completed")

        # Step 2: Data Transformation
        logging.info("Running data transformation step")
        transform_data()
        logging.info("Data transformation completed")

        # Step 3: Model Training
        logging.info("Running model training step")
        train_models()
        logging.info("Model training completed")

    except Exception as e:
        logging.error(f"An error occurred in the training pipeline: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
