import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. 页面配置
# ---------------------------------------------------------
st.set_page_config(page_title="AI 交易决策 Pro", layout="wide")
st.title("🤖 AI 交易决策 Pro：深度学习版")
st.caption("基于 Kraken 交易所数据 | 随机森林算法 | 支持长周期回测")

# ---------------------------------------------------------
# 2. 连接交易所 (Kraken)
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
# 3. 数据处理 (含分页循环下载逻辑)
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe, limit=3000):
    # 显示进度提示
    progress_text = f"正在挖掘 {symbol} 过去 {limit} 根 K 线的数据..."
    my_bar = st.progress(0, text=progress_text)
    
    # --- 分页下载逻辑 ---
    all_ohlcv = []
    
    # 1. 计算起始时间 (since)
    # 简单估算：1h=3600s, 4h=14400s
    timeframe_map = {'1h': 3600, '4h': 14400, '1d': 86400}
    duration_seconds = timeframe_map.get(timeframe, 3600)
    
    # 计算 limit 根 K 线大概需要回溯多少毫秒
    span_ms = limit * duration_seconds * 1000
    current_time = exchange.milliseconds()
    since = current_time - span_ms
    
    # 2. 循环抓取
    while len(all_ohlcv) < limit:
        try:
            # 每次请求 720 条 (Kraken 单次上限通常是 720)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=720)
            
            if not ohlcv:
                break # 如果没数据了，就停止
            
            # 更新时间指针：从刚才抓到的最后一条数据的下一秒开始抓
            last_time = ohlcv[-1][0]
            if last_time == since:
                break # 防止死循环
                
            since = last_time + 1 
            all_ohlcv += ohlcv
            
            # 更新进度条
            percent = min(len(all_ohlcv) / limit, 1.0)
            my_bar.progress(percent, text=f"{progress_text} ({len(all_ohlcv)}/{limit})")
            
            # 稍微休息一下，对交易所温柔一点
            time.sleep(0.2)
            
            # 如果已经抓到最新的数据了，就提前结束
            if last_time >= current_time - duration_seconds * 1000:
                break
                
        except Exception as e:
            st.warning(f"下载过程遇到小颠簸: {e}")
            break
    
    my_bar.empty() # 下载完清除进度条
    
    # 截取需要的长度 (因为可能多抓了一点)
    all_ohlcv = all_ohlcv[-limit:]
    
    if len(all_ohlcv) == 0:
        st.error("未获取到任何数据，请检查交易对名称 (例如 DOGE/USD) 或网络状态。")
        st.stop()

    # 转成 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
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

    # ATR (用于止盈止损)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Target (1=涨, 0=跌)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------
# 4. 训练与预测
# ---------------------------------------------------------
def train_and_predict(df):
    feature_cols = ['RSI', 'dist_SMA_7', 'returns', 'range', 'volume', 'ATR']
    X = df[feature_cols]
    y = df['Target']
    
    # 切分训练集
    split = int(len(X) * 0.8)
    # 增加树的数量到 200，让模型更稳
    model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    
    # 验证集准确率
    acc = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    
    # 预测最新
    latest = X.iloc[[-1]]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]
    
    return model, acc, pred, prob

# ---------------------------------------------------------
# 5. 主界面逻辑
# ---------------------------------------------------------
st.sidebar.header("控制面板")
symbol = st.sidebar.text_input("交易对", "DOGE/USD")
timeframe = st.sidebar.selectbox("周期", ['1h', '4h', '1d'])
# 增加一个数据量滑块，让你自己控制
limit_num = st.sidebar.slider("学习 K 线数量", 500, 5000, 3000, step=100)

if st.button("开始分析", type="primary"):
    try:
        df = fetch_and_prepare_data(symbol, timeframe, limit=limit_num)
        
        # 显示数据量确认
        st.toast(f"成功获取 {len(df)} 条有效数据，正在训练 AI...", icon="🧠")
        
        model, acc, pred, prob = train_and_predict(df)
        
        # 获取最新价格数据
        last_close = df['close'].iloc[-1]
        last_atr = df['ATR'].iloc[-1]
        
        # 计算策略建议
        stop_loss = 0.0
        take_profit = 0.0
        
        if pred == 1:
            direction = "做多 (Long) 🟢"
            stop_loss = last_close - (2.0 * last_atr)
            take_profit = last_close + (3.0 * last_atr)
        else:
            direction = "做空 (Short) 🔴"
            stop_loss = last_close + (2.0 * last_atr)
            take_profit = last_close - (3.0 * last_atr)

        # --- 展示结果 ---
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("当前价格", f"${last_close:.5f}")
        c2.metric("AI 建议方向", direction)
        c3.metric("信心指数", f"{prob[pred]*100:.1f}%")
        
        st.info(f"📊 **策略建议** (基于 {len(df)} 根 K 线训练)")
        c4, c5, c6 = st.columns(3)
        c4.metric("🛑 止损价格", f"${stop_loss:.5f}")
        c5.metric("🎯 止盈价格", f"${take_profit:.5f}")
        c6.metric("历史验证准确率", f"{acc*100:.1f}%")

        # 画图
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
        fig.add_hline(y=take_profit, line_dash="dash", line_color="green", annotation_text="止盈 TP")
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", annotation_text="止损 SL")
        
        fig.update_layout(height=500, title=f"{symbol} 走势预测图", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander(f"查看 {len(df)} 条原始数据"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"发生未知错误: {e}")
