import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

def main(args):
    assert os.path.exists(args.input), f"Processed file not found: {args.input}"
    assert os.path.exists(args.model), f"Model not found: {args.model}"
    assert os.path.exists("models/x_scaler.pkl") and os.path.exists("models/y_scaler.pkl"), "Scalers not found."

    df = pd.read_csv(args.input)
    X = df[["Timestamp", "Latitude", "Longitude"]].values
    y = df[["Magnitude", "Depth"]].values

    # Same split to replicate test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_scaler = joblib.load("models/x_scaler.pkl")
    y_scaler = joblib.load("models/y_scaler.pkl")

    X_test_s = x_scaler.transform(X_test)

    model = tf.keras.models.load_model(args.model)
    y_pred_s = model.predict(X_test_s)
    y_pred = y_scaler.inverse_transform(y_pred_s)

    mae_mag = mean_absolute_error(y_test[:,0], y_pred[:,0])
    rmse_mag = np.sqrt(mean_squared_error(y_test[:,0], y_pred[:,0]))
    mae_dep = mean_absolute_error(y_test[:,1], y_pred[:,1])
    rmse_dep = np.sqrt(mean_squared_error(y_test[:,1], y_pred[:,1],))

    print("=== Evaluation on Test Split ===")
    print(f"MAE Magnitude: {mae_mag:.3f} | RMSE Magnitude: {rmse_mag:.3f}")
    print(f"MAE Depth    : {mae_dep:.3f} | RMSE Depth    : {rmse_dep:.3f}")

    # After running the evaluation, save metrics to a text file
    os.makedirs("assets", exist_ok=True)

    # Save metrics in a text file
    with open("assets/metrics.txt", "w") as f:
        f.write(f"RMSE Magnitude: {rmse_mag:.4f}\n")
        f.write(f"RMSE Depth: {rmse_depth:.4f}\n")
        f.write(f"MAE Magnitude: {mae_mag:.4f}\n")
        f.write(f"MAE Depth: {mae_depth:.4f}\n")

    print("✅ Metrics saved at assets/metrics.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed.csv")
    parser.add_argument("--model", default="models/earth_model.h5")
    main(parser.parse_args())

