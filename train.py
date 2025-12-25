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
st.set_page_config(page_title="AI 自由版", layout="wide")
st.title("🤖 AI 交易决策：自由探索版")
st.caption("支持 BTC/ETH 快捷切换 | 支持手动输入任意币种 (Yahoo源)")

# ---------------------------------------------------------
# 2. 核心数据逻辑
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe, limit=3000):
    # 简单的格式清洗：把用户可能输入的 / 换成 -
    clean_symbol = symbol.strip().upper().replace("/", "-").replace("_", "-")
    
    period = "730d" 
    if timeframe == "1d":
        period = "max"
    
    st.toast(f"正在获取 {clean_symbol} 最近 {limit} 根 K 线...", icon="⚡")
    
    try:
        # 下载数据
        df = yf.download(tickers=clean_symbol, period=period, interval=timeframe, progress=False, multi_level_index=False)
        
        if df.empty:
            st.error(f"❌ 找不到交易对 {clean_symbol}。请去 Yahoo Finance 确认代码，通常格式为 COIN-USD (例如 BNB-USD, PEPE-USD)。")
            st.stop()
            
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        })
        
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        df = df.reset_index()
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

        # 特征工程
        df['returns'] = df['close'].pct_change()
        df['range'] = (df['high'] - df['low']) / df['close']
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['SMA_7'] = df['close'].rolling(7).mean()
        df['dist_SMA_7'] = (df['close'] - df['SMA_7']) / df['SMA_7']

        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        df.dropna(inplace=True)
        return df, clean_symbol

    except Exception as e:
        st.error(f"数据下载出错: {e}")
        st.stop()

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
# 3. 混合控制面板 (核心修改部分)
# ---------------------------------------------------------
st.sidebar.header("控制面板")

# 选项列表
options = [
    "BTC (比特币)", 
    "ETH (以太坊)", 
    "SOL (索拉纳)", 
    "DOGE (狗狗币)", 
    "🔍 自定义 (手动输入)"  # 新增选项
]

# 预设的映射关系
coin_map_presets = {
    "BTC (比特币)": "BTC-USD",
    "ETH (以太坊)": "ETH-USD",
    "SOL (索拉纳)": "SOL-USD",
    "DOGE (狗狗币)": "DOGE-USD"
}

# 推荐 K 线数量
recommendations = {
    "BTC-USD": 6000,
    "ETH-USD": 6000,
    "SOL-USD": 5000,
    "DOGE-USD": 3000
}

# 回调函数：处理滑块自动跳转
def update_slider():
    label = st.session_state.coin_selector
    # 只有选了预设币种时，才自动改滑块。选自定义时保持不变。
    if label in coin_map_presets:
        sym = coin_map_presets[label]
        rec_val = recommendations.get(sym, 3000)
        st.session_state.kline_slider = rec_val

st.sidebar.subheader("1. 选币模式")
selected_label = st.sidebar.radio(
    "请选择:", 
    options, 
    key="coin_selector", 
    on_change=update_slider
)

# 逻辑判断：是选了预设，还是自定义？
if selected_label == "🔍 自定义 (手动输入)":
    # 显示输入框
    user_input = st.sidebar.text_input(
        "请输入代码 (例如 BNB-USD, PEPE-USD):", 
        value="BNB-USD"
    )
    symbol = user_input
    display_name = user_input.upper() # 用于展示
else:
    # 使用预设
    symbol = coin_map_presets[selected_label]
    display_name = selected_label

st.sidebar.subheader("2. 参数设置")
timeframe = st.sidebar.selectbox("周期", ['1h', '1d'])

if 'kline_slider' not in st.session_state:
    st.session_state.kline_slider = 6000

limit_num = st.sidebar.slider(
    "K线数量", 
    500, 10000, 
    key="kline_slider",
    step=100
)

# ---------------------------------------------------------
# 4. 执行分析
# ---------------------------------------------------------
if st.button("开始分析", type="primary"):
    # 获取数据 (返回 df 和 清洗后的 symbol)
    df, real_symbol = fetch_and_prepare_data(symbol, timeframe, limit=limit_num)
    
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

    st.divider()
    st.subheader(f"{real_symbol} 分析报告")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("当前价格", f"${last_close:.6f}") # 改成6位小数以适应PEPE等小币
    c2.metric("AI 建议", direction)
    c3.metric("信心指数", f"{prob[pred]*100:.1f}%")
    
    st.info(f"📊 **策略建议** (基于 {len(df)} 根 K 线)")
    c4, c5, c6 = st.columns(3)
    c4.metric("🛑 止损 (SL)", f"${stop_loss:.6f}")
    c5.metric("🎯 止盈 (TP)", f"${take_profit:.6f}")
    c6.metric("验证准确率", f"{acc*100:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
    fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="止盈")
    fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="止损")
    fig.update_layout(height=500, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
