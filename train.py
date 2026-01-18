# Copyright 2023, MetaQuotes Ltd.
# https://www.mql5.com

# python libraries
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx
import os 
import shutil
import subprocess
from datetime import timedelta, datetime
import sklearn
from sklearn.preprocessing import MinMaxScaler

print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"tf2onnx: {tf2onnx.__version__}")

# Lista de símbolos a procesar
symbols = [ "USDCAD", "EURUSD","GBPUSD", "USDCHF", "AUDUSD"]

# input parameters
inp_history_size = 120

# Rutas para leer CSVs del directorio raíz y guardar modelos
data_path = "./"  # Directorio raíz donde están los CSVs
file_path = "./models/"  # Carpeta donde se guardarán los ONNX

# Crear carpeta models si no existe
os.makedirs(file_path, exist_ok=True)

print("data path to read CSVs:", data_path)
print("file path to save onnx model:", file_path)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
       # find the end of this pattern
       end_ix = i + n_steps
       # check if we are beyond the sequence
       if end_ix > len(sequence)-1:
          break
       # gather input and output parts of the pattern
       seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
       X.append(seq_x)
       y.append(seq_y)
    return np.array(X), np.array(y)

# Function to train and save a model with given dates and model name
def train_and_save_model(simbol, end_date, model_suffix=""):
    """
    Trains a model for a given symbol and time period.
    
    Parameters:
    - simbol: Trading symbol (e.g., "EURUSD")
    - end_date: End date for the training data
    - model_suffix: Suffix to add to model name (e.g., "backtesting")
    """
    # Calculate start date based on end date
    start_date = end_date - timedelta(days=inp_history_size)
    
    # Create model name with optional suffix
    if model_suffix:
        inp_model_name = f"model.{simbol}.H1.120.{model_suffix}.onnx"
    else:
        inp_model_name = f"model.{simbol}.H1.120.onnx"
    
    print(f"\n{'='*60}")
    print(f"Procesando {simbol} - {model_suffix if model_suffix else 'Trading'}")
    print(f"Período: {start_date} hasta {end_date}")
    print(f"{'='*60}\n")
    
    try:
        # Leer datos desde CSV en directorio raíz
        csv_file = f"{data_path}{simbol}_H1.csv"
        
        if not os.path.exists(csv_file):
            print(f"ERROR: No se encontró el archivo {csv_file}")
            return False
        
        # Leer CSV con separador de tabulación
        df = pd.read_csv(csv_file, sep='\t')
        
        # Convertir columna DATE y TIME a datetime
        df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S')
        
        # Filtrar por rango de fechas
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
        
        if len(df) == 0:
            print(f"ERROR: No hay datos en el rango de fechas para {simbol}")
            return False
        
        print(f"Registros obtenidos: {len(df)}")
        
        # get close prices only
        data = df[['<CLOSE>']].values
        
        # scale data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        
        # training size is 80% of the data
        training_size = int(len(scaled_data)*0.80) 
        print(f"Training_size: {training_size}")
        train_data_initial = scaled_data[0:training_size,:]
        test_data_initial = scaled_data[training_size:,:1]
        
        # split into samples
        time_step = inp_history_size
        x_train, y_train = split_sequence(train_data_initial, time_step)
        x_test, y_test = split_sequence(test_data_initial, time_step)
        
        # reshape input to be [samples, time steps, features] which is required for LSTM
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        # define model
        from keras.models import Sequential
        from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
        from keras.metrics import RootMeanSquaredError as rmse
        from keras.callbacks import EarlyStopping
        
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(inp_history_size,1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mse', metrics=[rmse()])
        
        # Configurar Early Stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        # model training con early stopping
        print(f"\nEntrenando modelo para {simbol} ({model_suffix if model_suffix else 'Trading'})...")
        history = model.fit(
            x_train, y_train, 
            epochs=300, 
            validation_data=(x_test, y_test), 
            batch_size=32, 
            callbacks=[early_stop],
            verbose=1
        )
        
        # evaluate training data
        train_loss, train_rmse = model.evaluate(x_train, y_train, batch_size=32)
        print(f"train_loss={train_loss:.3f}")
        print(f"train_rmse={train_rmse:.3f}")
        
        # evaluate testing data
        test_loss, test_rmse = model.evaluate(x_test, y_test, batch_size=32)
        print(f"test_loss={test_loss:.3f}")
        print(f"test_rmse={test_rmse:.3f}")
        
        # First export as TensorFlow SavedModel
        temp_model_path = f"/tmp/temp_model_{simbol}_{model_suffix if model_suffix else 'trading'}"
        
        # Remove temp directory if exists
        if os.path.exists(temp_model_path):
            shutil.rmtree(temp_model_path)
        
        # Export model
        model.export(temp_model_path)
        
        # Convert to ONNX using command line tool
        output_path = file_path + inp_model_name
        cmd = f'python -m tf2onnx.convert --saved-model "{temp_model_path}" --output "{output_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Modelo guardado en: {output_path}")
        else:
            print(f"✗ Error guardando modelo: {result.stderr}")
        
        # Clean up temp directory
        shutil.rmtree(temp_model_path)
        
        print(f"\n✓ {simbol} - {model_suffix if model_suffix else 'Trading'} completado exitosamente!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR procesando {simbol} - {model_suffix if model_suffix else 'Trading'}: {str(e)}\n")
        return False

# Procesar cada símbolo con dos modelos
for simbol in symbols:
    print(f"\n{'#'*60}")
    print(f"# INICIANDO PROCESAMIENTO DE {simbol}")
    print(f"{'#'*60}\n")
    
    # Modelo 1: Trading (fecha actual)
    end_date_trading = datetime.now()
    success_trading = train_and_save_model(simbol, end_date_trading, model_suffix="")
    
    # Modelo 2: Backtesting (hace 7 días)
    end_date_backtesting = datetime.now() - timedelta(days=7)
    success_backtesting = train_and_save_model(simbol, end_date_backtesting, model_suffix="backtesting")
    
    if success_trading and success_backtesting:
        print(f"\n{'='*60}")
        print(f"✓✓ AMBOS MODELOS PARA {simbol} COMPLETADOS EXITOSAMENTE")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"✗ ALGUNOS MODELOS PARA {simbol} FALLARON")
        print(f"{'='*60}\n")

print(f"\n{'#'*60}")
print("# PROCESO COMPLETADO PARA TODOS LOS SÍMBOLOS")
print(f"{'#'*60}")
