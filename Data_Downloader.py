import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os

# ===================== CONFIG =====================
SYMBOLS = [
    "USDCAD",
    "USDCHF",
    "USDJPY",
    "GBPUSD",
    "EURUSD",
    "EURGBP",
    "GBPJPY"
]

TIMEFRAME = mt5.TIMEFRAME_H1
DAYS = 120
#OUTPUT_DIR = "mt5_data_h1"
OUTPUT_DIR = "mt5_data_h1_backtesting"
# =================================================

# Crear carpeta de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar MT5
if not mt5.initialize():
    raise RuntimeError(f"MT5 no pudo inicializarse → {mt5.last_error()}")

timezone = pytz.UTC
#date_from = datetime.now(timezone) - timedelta(days=DAYS)
date_from = datetime(2025, 12, 1, tzinfo=timezone)- timedelta(days=DAYS)

for symbol in SYMBOLS:
    print(f"📥 Descargando {symbol}...")

    # Activar símbolo
    if not mt5.symbol_select(symbol, True):
        print(f"❌ No se pudo seleccionar {symbol}")
        continue

    rates = mt5.copy_rates_from(
        symbol,
        TIMEFRAME,
        date_from,
        DAYS * 24
    )

    if rates is None or len(rates) == 0:
        print(f"❌ Sin datos para {symbol}")
        continue

    df = pd.DataFrame(rates)

    # Convertir tiempo
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Guardar CSV en minúsculas
    filename = f"{symbol.lower()}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)

    print(f"✅ Guardado: {filepath}")

mt5.shutdown()
print("🎯 Descarga finalizada.")
