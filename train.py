#!/usr/bin/env python3
# ===================================================================
# SCRIPT DE ENTRENAMIENTO FOREX - GitHub Codespaces
# ===================================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import os 
import sys
import shutil
import subprocess
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
import tf2onnx

# Importar Keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.metrics import RootMeanSquaredError as rmse
from keras.callbacks import EarlyStopping

print("✓ Librerías importadas")
print(f"TensorFlow: {tf.__version__}")
print(f"Python: {sys.version}\n")

# ===================================================================
# CONFIGURACIÓN
# ===================================================================
DATA_PATH = './'  # CSVs en el mismo directorio
MODEL_PATH = './models/'  # Carpeta para ONNX

os.makedirs(MODEL_PATH, exist_ok=True)

symbols = ["USDCAD", "EURUSD", "GBPUSD", "USDCHF", "AUDUSD"]
inp_history_size = 120
DAYS_TO_USE = 120

# ===================================================================
# FUNCIONES
# ===================================================================

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
       end_ix = i + n_steps
       if end_ix > len(sequence)-1:
          break
       seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
       X.append(seq_x)
       y.append(seq_y)
    return np.array(X), np.array(y)

def train_and_save_model(simbol, end_date, model_suffix=""):
    start_date = end_date - timedelta(days=DAYS_TO_USE)
    
    if model_suffix:
        inp_model_name = f"model.{simbol}.H1.120.{model_suffix}.onnx"
    else:
        inp_model_name = f"model.{simbol}.H1.120.onnx"
    
    print(f"\n{'='*60}")
    print(f"Procesando {simbol} - {model_suffix if model_suffix else 'Trading'}")
    print(f"Período: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")
    
    try:
        csv_file = f"{DATA_PATH}{simbol}_H1.csv"
        
        if not os.path.exists(csv_file):
            print(f"❌ No se encontró {csv_file}")
            return False
        
        df = pd.read_csv(csv_file, sep='\t')
        df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        if len(df) == 0:
            print(f"❌ No hay datos en el rango")
            return False
        
        print(f"✓ {len(df)} registros")
        
        data = df[['<CLOSE>']].values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        
        training_size = int(len(scaled_data)*0.80)
        train_data = scaled_data[0:training_size,:]
        test_data = scaled_data[training_size:,:1]
        
        x_train, y_train = split_sequence(train_data, inp_history_size)
        x_test, y_test = split_sequence(test_data, inp_history_size)
        
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        # Modelo
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(inp_history_size,1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=[rmse()])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
        
        print("🚀 Entrenando...")
        model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test), 
                  batch_size=32, callbacks=[early_stop], verbose=2)
        
        train_loss, train_rmse = model.evaluate(x_train, y_train, batch_size=32, verbose=0)
        test_loss, test_rmse = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
        print(f"📈 Train: loss={train_loss:.3f}, rmse={train_rmse:.3f}")
        print(f"📉 Test: loss={test_loss:.3f}, rmse={test_rmse:.3f}")
        
        # Exportar
        temp_path = f"/tmp/temp_{simbol}_{model_suffix if model_suffix else 'trading'}"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        
        model.export(temp_path)
        
        output_path = MODEL_PATH + inp_model_name
        cmd = [sys.executable, '-m', 'tf2onnx.convert', 
               '--saved-model', temp_path, '--output', output_path, '--opset', '13']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        
        if result.returncode == 0:
            print(f"✅ Guardado: {output_path}\n")
            return True
        else:
            print(f"❌ Error: {result.stderr}\n")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False

# ===================================================================
# MAIN
# ===================================================================

print(f"\n{'#'*60}")
print("# ENTRENAMIENTO DE MODELOS FOREX")
print(f"{'#'*60}\n")

for simbol in symbols:
    print(f"\n### {simbol} ###")
    
    end_trading = datetime.now()
    end_backtest = datetime.now() - timedelta(days=7)
    
    ok1 = train_and_save_model(simbol, end_trading, "")
    ok2 = train_and_save_model(simbol, end_backtest, "backtesting")
    
    if ok1 and ok2:
        print(f"✅ {simbol} completado")
    else:
        print(f"⚠️ {simbol} con errores")

print(f"\n{'#'*60}")
print("# ✅ PROCESO COMPLETADO")
print(f"{'#'*60}")
print(f"Modelos en: {MODEL_PATH}")
