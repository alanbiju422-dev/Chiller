import os
import json
import logging
from data.loader import load_and_prepare_data
from ml.preprocessing import Preprocessor
from ml.features import create_features
from ml.expected_energy import train_and_select_model
from ml.anomaly_detector import ContextualAnomalyDetector
from ml.artifacts import get_artifact_path
from ml.evaluator import evaluate_predictions
import pandas as pd

logging.basicConfig(level=logging.INFO)

def run_training_pipeline():
    logging.info("Starting Data Loading...")
    train_df, unseen_df = load_and_prepare_data('development_dataset.csv')
    
    logging.info(f"Train Size: {len(train_df)}, Unseen Size: {len(unseen_df)}")
    
    logging.info("Feature Engineering...")
    X_train = create_features(train_df)
    y_train = train_df['Chiller Energy Consumption (kWh)']
    
    X_unseen = create_features(unseen_df)
    y_unseen = unseen_df['Chiller Energy Consumption (kWh)']
    
    logging.info("Preprocessing...")
    preprocessor = Preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_unseen_proc = preprocessor.transform(X_unseen)
    
    logging.info("Training Expected Energy Model...")
    energy_model = train_and_select_model(X_train_proc, y_train)
    
    logging.info("Training Anomaly Detector (Isolation Forest)...")
    anomaly_detector = ContextualAnomalyDetector()
    anomaly_detector.train(X_train_proc)
    
    logging.info("Evaluating Unseen Data...")
    preds_unseen = energy_model.predict(X_unseen_proc)
    metrics = evaluate_predictions(y_unseen, preds_unseen)
    logging.info(f"Final Unseen Metrics: {metrics}")
    
    logging.info("Saving Artifacts...")
    preprocessor.save(get_artifact_path('preprocessor.joblib'))
    energy_model.save(get_artifact_path('energy_model.joblib'))
    anomaly_detector.save(get_artifact_path('anomaly_detector.joblib'))
    
    # Save unseen data for replay
    os.makedirs('data_cache', exist_ok=True)
    unseen_df.to_pickle('data_cache/unseen_data.pkl')
    
    # Save metrics
    with open(get_artifact_path('metrics.json'), 'w') as f:
        json.dump({
            'train_size': len(train_df),
            'unseen_size': len(unseen_df),
            'metrics': metrics
        }, f, indent=4)
        
    logging.info("Training Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_training_pipeline()
