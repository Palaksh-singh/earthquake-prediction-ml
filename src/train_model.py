import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

def build_model(input_dim: int):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="linear")  # Magnitude, Depth (regression)
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def main(args):
    assert os.path.exists(args.input), f"Processed file not found: {args.input}"
    df = pd.read_csv(args.input)

    X = df[["Timestamp", "Latitude", "Longitude"]].values
    y = df[["Magnitude", "Depth"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_test_s  = x_scaler.transform(X_test)
    y_train_s = y_scaler.fit_transform(y_train)
    y_test_s  = y_scaler.transform(y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(x_scaler, "models/x_scaler.pkl")
    joblib.dump(y_scaler, "models/y_scaler.pkl")

    model = build_model(X_train_s.shape[1])

    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_test_s, y_test_s),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=[es]
    )

    # --- Save Training History Plot ---

    os.makedirs("assets/screenshots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/screenshots/loss_curve.png")
    plt.close()

    print("✅ Training loss curve saved at assets/screenshots/loss_curve.png")


    # Evaluate (invert scaling)
    y_pred_s = model.predict(X_test_s)
    y_pred = y_scaler.inverse_transform(y_pred_s)

    mae_mag = mean_absolute_error(y_test[:,0], y_pred[:,0])
    import numpy as np
    rmse_mag = np.sqrt(mean_squared_error(y_test[:,0], y_pred[:,0]))
    mae_dep = mean_absolute_error(y_test[:,1], y_pred[:,1])
    rmse_dep = np.sqrt(mean_squared_error(y_test[:,1], y_pred[:,1]))

    print(f"MAE Magnitude: {mae_mag:.3f} | RMSE Magnitude: {rmse_mag:.3f}")
    print(f"MAE Depth    : {mae_dep:.3f} | RMSE Depth    : {rmse_dep:.3f}")

    # Save model
    model.save(args.model)
    print(f"[OK] Model saved to {args.model}")

    # Plot training history
    os.makedirs("assets", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.title("Training History"); plt.legend()
    plt.tight_layout()
    plt.savefig("assets/training_loss.png", dpi=150)
    print("[OK] Training plot saved to assets/training_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed.csv")
    parser.add_argument("--model", default="models/earth_model.h5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    main(parser.parse_args())
