import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. 页面配置
# ---------------------------------------------------------
st.set_page_config(page_title="AI 交易决策Pro", layout="wide")
st.title("🤖 AI 交易决策 Pro：带止盈止损版")

# ---------------------------------------------------------
# 2. 连接交易所
# ---------------------------------------------------------
@st.cache_resource
def init_exchange():
    return ccxt.kraken({'enableRateLimit': True})

try:
    exchange = init_exchange()
except Exception as e:
    st.error(f"连接失败: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 数据处理 (新增 ATR 计算)
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe):
    st.toast(f"正在分析 {symbol}...", icon="🔍")
    
    # 获取数据
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # --- 特征工程 ---
    df['returns'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 均线
    df['SMA_7'] = df['close'].rolling(7).mean()
    df['dist_SMA_7'] = (df['close'] - df['SMA_7']) / df['SMA_7']

    # --- 新增：ATR (波动率) 计算，用于止盈止损 ---
    # TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 目标 (Label)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------
# 4. 训练与策略
# ---------------------------------------------------------
def train_and_predict(df):
    feature_cols = ['RSI', 'dist_SMA_7', 'returns', 'range', 'volume', 'ATR']
    X = df[feature_cols]
    y = df['Target']
    
    # 训练模型
    split = int(len(X) * 0.8)
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    
    # 准确率
    acc = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    
    # 预测未来
    latest = X.iloc[[-1]]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]
    
    return model, acc, pred, prob

# ---------------------------------------------------------
# 5. 主界面
# ---------------------------------------------------------
st.sidebar.header("控制面板")
symbol = st.sidebar.text_input("交易对", "DOGE/USD")
timeframe = st.sidebar.selectbox("周期", ['1h', '4h', '1d'])

if st.button("开始分析", type="primary"):
    try:
        df = fetch_and_prepare_data(symbol, timeframe)
        model, acc, pred, prob = train_and_predict(df)
        
        # 获取最新价格和ATR
        last_close = df['close'].iloc[-1]
        last_atr = df['ATR'].iloc[-1]
        
        # --- 计算止盈止损 (策略：2倍ATR止损，3倍ATR止盈) ---
        stop_loss = 0.0
        take_profit = 0.0
        
        if pred == 1: # AI 看涨
            direction = "做多 (Long) 🟢"
            stop_loss = last_close - (2.0 * last_atr)
            take_profit = last_close + (3.0 * last_atr)
            signal_color = "green"
        else: # AI 看跌
            direction = "做空 (Short) 🔴"
            stop_loss = last_close + (2.0 * last_atr)
            take_profit = last_close - (3.0 * last_atr)
            signal_color = "red"

        # --- 显示结果 ---
        st.subheader(f"{symbol} 交易建议")
        
        # 第一行：核心信号
        c1, c2, c3 = st.columns(3)
        c1.metric("当前价格", f"${last_close:.4f}")
        c2.metric("AI 方向", direction, delta_color="off")
        c3.metric("信心指数", f"{prob[pred]*100:.1f}%")
        
        # 第二行：止盈止损卡片
        st.info("📊 **基于 ATR 波动率的建议点位** (盈亏比 1.5:1)")
        c4, c5, c6 = st.columns(3)
        c4.metric("🛑 建议止损 (SL)", f"${stop_loss:.4f}", help="触碰此价格必须离场，防止亏损扩大")
        c5.metric("🎯 建议止盈 (TP)", f"${take_profit:.4f}", help="触碰此价格落袋为安")
        c6.metric("历史准确率", f"{acc*100:.1f}%", help="如果低于50%，建议反着做或观望")

        # 图表
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
        
        # 在图上画出止盈止损线
        fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="止盈 TP")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="止损 SL")
        
        fig.update_layout(height=500, title=f"{symbol} 价格走势图")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"出错: {e}")
