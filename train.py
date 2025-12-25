import os
import subprocess
import sys
import time

# --- 自动安装运行环境 (专治各种 ModuleNotFound) ---
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 这里列出你所有的库
install_package("ccxt")
install_package("pandas")
install_package("plotly")
install_package("scikit-learn")
# ---------------------------------------------

import streamlit as st
import ccxt
import pandas as pd

st.title("我的量化交易机器人")
st.write("环境检查完毕，ccxt 已成功安装！")

# ↓↓↓↓ 把你原来的代码从这里开始粘贴 ↓↓↓↓
# ...import ccxt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === 这里是我们手写的计算公式，替代 pandas_ta ===
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()
# ============================================

print("1. 正在从币安下载 ETH 历史数据...")
exchange = ccxt.binance()
try:
    # 获取数据
    bars = exchange.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=1000)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    print("2. 正在整理数据特征...")
    # 使用我们要上面手写的函数来计算
    df['RSI'] = calculate_rsi(df['close'], 14)
    df['EMA_20'] = calculate_ema(df['close'], 20)
    df['Return'] = df['close'].pct_change()
    
    # 设定目标
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)

    # 准备训练
    features = ['RSI', 'EMA_20', 'Return', 'volume']
    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("3. AI 开始学习...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"✅ 训练完成！准确率: {accuracy_score(y_test, preds)*100:.2f}%")
    
    model.save_model("my_crypto_model.json")
    print("💾 模型已保存为 my_crypto_model.json")

except Exception as e:

    print(f"出错啦: {e}")
