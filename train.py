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
st.set_page_config(page_title="AI 交易决策 (雅虎版)", layout="wide")
st.title("🤖 AI 交易决策 Pro：雅虎财经版")
st.caption("数据源：Yahoo Finance | 无限制下载 | 极速响应")

# ---------------------------------------------------------
# 2. 数据处理 (使用 yfinance，超级稳定)
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe, limit=3000):
    # 雅虎财经的格式转换：把 DOGE/USD 转换成 DOGE-USD
    yahoo_symbol = symbol.replace("/", "-").replace("_", "-")
    
    # 雅虎的时间周期映射
    # limit 在这里主要用来控制回溯天数，因为雅虎是按“天”下载的
    period = "730d" # 默认下载最近2年数据 (1h数据的上限通常是730天)
    if timeframe == "1d":
        period = "max" # 日线可以无限长
    
    st.toast(f"正在从雅虎财经下载 {yahoo_symbol} 数据...", icon="📥")
    
    try:
        # 下载数据 (自动修复多级索引问题)
        df = yf.download(tickers=yahoo_symbol, period=period, interval=timeframe, progress=False, multi_level_index=False)
        
        if df.empty:
            st.error(f"❌ 找不到交易对 {yahoo_symbol}，请检查拼写 (例如尝试 DOGE-USD 或 BTC-USD)")
            st.stop()
            
        # 雅虎下载的列名首字母是大写的，统一一下
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low", 
            "Close": "close", "Volume": "volume"
        })
        
        # 只要最后 limit 条
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        # 必须重置索引，把 Date/Datetime 变成一列
        df = df.reset_index()
        # 雅虎的日期列名可能是 'Date' 或 'Datetime'
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})

        # 去除时区信息 (避免绘图报错)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

        # --- 特征工程 (保持不变) ---
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
        st.error(f"雅虎数据下载失败: {e}")
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
# 4. 主界面
# ---------------------------------------------------------
st.sidebar.header("控制面板")
# 雅虎财经代码习惯：BTC-USD, DOGE-USD, ETH-USD
symbol = st.sidebar.text_input("交易对", "DOGE-USD") 
# 雅虎的 1小时数据很稳，但雅虎不支持 4h 数据，所以只提供 1h 和 1d
timeframe = st.sidebar.selectbox("周期", ['1h', '1d'])
limit_num = st.sidebar.slider("学习 K 线数量", 500, 10000, 3000, step=100)

if st.button("开始分析", type="primary"):
    # 获取数据
    df = fetch_and_prepare_data(symbol, timeframe, limit=limit_num)
    
    st.toast(f"成功获取 {len(df)} 条数据，AI 正在计算...", icon="🧠")
    
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

    # 展示
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("当前价格", f"${last_close:.5f}")
    c2.metric("AI 建议", direction)
    c3.metric("信心指数", f"{prob[pred]*100:.1f}%")
    
    st.info(f"📊 **策略建议** (基于雅虎财经 {len(df)} 条数据)")
    c4, c5, c6 = st.columns(3)
    c4.metric("🛑 止损 (SL)", f"${stop_loss:.5f}")
    c5.metric("🎯 止盈 (TP)", f"${take_profit:.5f}")
    c6.metric("验证准确率", f"{acc*100:.1f}%")

    # 画图
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
    fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="止盈")
    fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="止损")
    fig.update_layout(height=500, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
