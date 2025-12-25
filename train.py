import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. 页面配置
# ---------------------------------------------------------
st.set_page_config(page_title="AI 极速版", layout="wide")
st.title("🚀 AI 交易决策：极速切换版")
st.caption("支持 BTC / ETH / SOL / DOGE 一键切换 | 数据源：Yahoo Finance")

# ---------------------------------------------------------
# 2. 数据处理 (雅虎财经源)
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe, limit=10000):
    # 这里的 symbol 已经是处理好的格式 (如 BTC-USD)
    
    period = "730d" # 默认下载2年
    if timeframe == "1d":
        period = "max"
    
    st.toast(f"正在获取 {symbol} 的最新数据...", icon="⚡")
    
    try:
        # 下载数据
        df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False, multi_level_index=False)
        
        if df.empty:
            st.error(f"无法获取数据，请稍后重试。")
            st.stop()
            
        # 统一列名
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        })
        
        # 截取需要的长度
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        # 处理索引
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})

        # 去除时区
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

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

        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Target
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        df.dropna(inplace=True)
        return df

    except Exception as e:
        st.error(f"数据下载出错: {e}")
        st.stop()

# ---------------------------------------------------------
# 3. 训练与预测
# ---------------------------------------------------------
def train_and_predict(df):
    feature_cols = ['RSI', 'dist_SMA_7', 'returns', 'range', 'volume', 'ATR']
    X = df[feature_cols]
    y = df['Target']
    
    split = int(len(X) * 0.8)
    model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    
    acc = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    
    latest = X.iloc[[-1]]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]
    
    return model, acc, pred, prob

# ---------------------------------------------------------
# 4. 主界面 (这里修改了！)
# ---------------------------------------------------------
st.sidebar.header("控制面板")

# --- 修改开始：使用单选按钮快速切换 ---
st.sidebar.subheader("1. 选择币种")
# 定义显示的名字和实际代码的对应关系
coin_map = {
    "BTC (比特币)": "BTC-USD",
    "ETH (以太坊)": "ETH-USD",
    "SOL (索拉纳)": "SOL-USD",
    "DOGE (狗狗币)": "DOGE-USD"
}
# 获取用户选择的中文名
selected_label = st.sidebar.radio("点击直接切换:", list(coin_map.keys()))
# 拿到实际的代码 (例如 DOGE-USD)
symbol = coin_map[selected_label]
# --- 修改结束 ---

st.sidebar.subheader("2. 参数设置")
timeframe = st.sidebar.selectbox("周期", ['1h', '1d'])
limit_num = st.sidebar.slider("K线数量 (建议 ETH 设大)", 500, 10000, 3000, step=100)

if st.button("开始分析", type="primary"):
    df = fetch_and_prepare_data(symbol, timeframe, limit=limit_num)
    
    model, acc, pred, prob = train_and_predict(df)
    
    last_close = df['close'].iloc[-1]
    last_atr = df['ATR'].iloc[-1]
    
    if pred == 1:
        direction = "做多 (Long) 🟢"
        stop_loss = last_close - (2.0 * last_atr)
        take_profit = last_close + (3.0 * last_atr)
    else:
        direction = "做空 (Short) 🔴"
        stop_loss = last_close + (2.0 * last_atr)
        take_profit = last_close - (3.0 * last_atr)

    # 结果展示
    st.divider()
    st.subheader(f"{selected_label} 分析结果") # 显示当前选中的币种
    
    c1, c2, c3 = st.columns(3)
    c1.metric("当前价格", f"${last_close:.4f}")
    c2.metric("AI 建议", direction)
    c3.metric("信心指数", f"{prob[pred]*100:.1f}%")
    
    st.info(f"📊 **策略建议** (基于 {len(df)} 根 K 线)")
    c4, c5, c6 = st.columns(3)
    c4.metric("🛑 止损 (SL)", f"${stop_loss:.4f}")
    c5.metric("🎯 止盈 (TP)", f"${take_profit:.4f}")
    c6.metric("验证准确率", f"{acc*100:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
    fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="止盈")
    fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="止损")
    fig.update_layout(height=500, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
