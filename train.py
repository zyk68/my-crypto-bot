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
st.set_page_config(page_title="AI 胜率强化版", layout="wide")
st.title("🤖 AI 交易决策：胜率强化版")
st.caption("新增 MACD + 布林带 + EMA200 趋势过滤 | 专为提高胜率设计")

# ---------------------------------------------------------
# 2. 数据处理 (特征工程大升级)
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe, limit=3000):
    clean_symbol = symbol.strip().upper().replace("/", "-").replace("_", "-")
    
    period = "730d" 
    if timeframe == "1d":
        period = "max"
    
    st.toast(f"正在深度分析 {clean_symbol} 的多维数据...", icon="🔬")
    
    try:
        df = yf.download(tickers=clean_symbol, period=period, interval=timeframe, progress=False, multi_level_index=False)
        
        if df.empty:
            st.error(f"❌ 找不到交易对 {clean_symbol}")
            st.stop()
            
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        df = df.reset_index()
        col_name = 'Date' if 'Date' in df.columns else 'Datetime'
        df = df.rename(columns={col_name: 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

        # --- 特征工程 2.0 (加入更多指标) ---
        
        # 1. 基础特征
        df['returns'] = df['close'].pct_change()
        df['range'] = (df['high'] - df['low']) / df['close']
        
        # 2. RSI (相对强弱)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD (趋势王牌)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal'] # 柱状图，这是 AI 重点学习的
        
        # 4. 布林带 (波动率位置)
        df['BB_Mid'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
        # 计算价格在布林带中的相对位置 (0=下轨, 1=上轨)
        df['BB_Pos'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 5. EMA 200 (牛熊分界线 - 用于趋势过滤)
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
        df['Dist_EMA200'] = (df['close'] - df['EMA_200']) / df['EMA_200'] # 距离EMA200有多远

        # 6. ATR (止盈止损用)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Target
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        df.dropna(inplace=True)
        return df, clean_symbol

    except Exception as e:
        st.error(f"数据出错: {e}")
        st.stop()

def train_and_predict(df):
    # AI 现在要看更多的“试卷题目”
    feature_cols = ['RSI', 'MACD', 'MACD_Hist', 'BB_Pos', 'Dist_EMA200', 'returns', 'volume', 'ATR']
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
# 3. 控制面板
# ---------------------------------------------------------
st.sidebar.header("控制面板")

options = ["BTC (比特币)", "ETH (以太坊)", "SOL (索拉纳)", "DOGE (狗狗币)", "🔍 自定义"]
coin_map_presets = {"BTC (比特币)": "BTC-USD", "ETH (以太坊)": "ETH-USD", "SOL (索拉纳)": "SOL-USD", "DOGE (狗狗币)": "DOGE-USD"}
recommendations = {"BTC-USD": 6000, "ETH-USD": 6000, "SOL-USD": 5000, "DOGE-USD": 3000}

def update_slider():
    label = st.session_state.coin_selector
    if label in coin_map_presets:
        sym = coin_map_presets[label]
        st.session_state.kline_slider = recommendations.get(sym, 3000)

st.sidebar.subheader("1. 选币模式")
selected_label = st.sidebar.radio("请选择:", options, key="coin_selector", on_change=update_slider)

if selected_label == "🔍 自定义":
    symbol = st.sidebar.text_input("输入代码 (如 PEPE-USD):", value="BNB-USD").upper()
    display_name = symbol
else:
    symbol = coin_map_presets[selected_label]
    display_name = selected_label

st.sidebar.subheader("2. 核心设置")
timeframe = st.sidebar.selectbox("周期", ['1h', '1d'])
if 'kline_slider' not in st.session_state: st.session_state.kline_slider = 6000
limit_num = st.sidebar.slider("K线数量", 500, 10000, key="kline_slider", step=100)

# --- 新增功能：趋势滤网 ---
st.sidebar.divider()
st.sidebar.subheader("3. 🛡️ 胜率保护盾")
use_trend_filter = st.sidebar.checkbox("开启 EMA200 趋势过滤 (强烈推荐)", value=True, help="勾选后，只有当价格在 EMA200 之上时才允许做多，之下时才允许做空。这能过滤掉大部分逆势亏损单。")

# ---------------------------------------------------------
# 4. 执行分析
# ---------------------------------------------------------
if st.button("开始深度分析", type="primary"):
    df, real_symbol = fetch_and_prepare_data(symbol, timeframe, limit=limit_num)
    model, acc, pred, prob = train_and_predict(df)
    
    last_close = df['close'].iloc[-1]
    last_atr = df['ATR'].iloc[-1]
    last_ema200 = df['EMA_200'].iloc[-1]
    
    # --- 核心逻辑：结合 AI 预测 + 趋势过滤 ---
    ai_direction = "多" if pred == 1 else "空"
    final_decision = "观望 (Wait)" # 默认观望
    reason = "AI 信心不足或趋势不符"
    color = "gray"
    
    # 1. 基础方向
    if pred == 1: # AI 看多
        if use_trend_filter and last_close < last_ema200:
            final_decision = "🚫 建议观望 (趋势过滤)"
            reason = f"AI 想做多，但价格低于 EMA200 (${last_ema200:.2f})，属于逆势接飞刀，风险极大。"
            color = "orange"
        else:
            final_decision = "做多 (Long) 🟢"
            reason = "AI 看涨，且符合大趋势 (或未开启过滤)。"
            color = "green"
            stop_loss = last_close - (2.0 * last_atr)
            take_profit = last_close + (3.0 * last_atr)
            
    else: # AI 看空
        if use_trend_filter and last_close > last_ema200:
            final_decision = "🚫 建议观望 (趋势过滤)"
            reason = f"AI 想做空，但价格高于 EMA200 (${last_ema200:.2f})，属于牛市摸顶，容易被套。"
            color = "orange"
        else:
            final_decision = "做空 (Short) 🔴"
            reason = "AI 看跌，且符合大趋势 (或未开启过滤)。"
            color = "red"
            stop_loss = last_close + (2.0 * last_atr)
            take_profit = last_close - (3.0 * last_atr)

    # 结果展示
    st.divider()
    st.subheader(f"{real_symbol} 深度分析报告")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("当前价格", f"${last_close:.6f}")
    c2.metric("最终决策", final_decision) # delta_color会被上面的逻辑覆盖，这里简单显示
    c3.metric("AI 原始信心", f"{prob[pred]*100:.1f}%")
    
    # 决策解释框
    if color == "green":
        st.success(f"💡 **执行建议**: {reason}")
    elif color == "red":
        st.error(f"💡 **执行建议**: {reason}")
    else:
        st.warning(f"💡 **执行建议**: {reason}")

    if "观望" not in final_decision:
        st.info(f"📊 **点位建议**")
        c4, c5, c6 = st.columns(3)
        c4.metric("🛑 止损 (SL)", f"${stop_loss:.6f}")
        c5.metric("🎯 止盈 (TP)", f"${take_profit:.6f}")
        c6.metric("历史准确率", f"{acc*100:.1f}%")

    # 画图
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
    # 画出 EMA200 线
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_200'], line=dict(color='purple', width=2), name='EMA 200 (牛熊线)'))
    
    if "观望" not in final_decision:
        fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="止盈")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="止损")
        
    fig.update_layout(height=500, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
