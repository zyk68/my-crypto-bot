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
st.set_page_config(page_title="AI 交易决策 Pro (修复版)", layout="wide")
st.title("🤖 AI 交易决策 Pro：强力数据版")
st.caption("已破解 720 条数据限制 | 自动分批拼接 | 支持深度学习")

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
# 3. 数据处理 (修复了只下载720条的问题)
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe, limit=3000):
    # 进度条
    progress_text = f"正在暴力挖掘 {symbol} 数据 (目标: {limit} 条)..."
    my_bar = st.progress(0, text=progress_text)
    
    all_ohlcv = []
    
    # --- 核心修复：更稳健的时间计算 ---
    # 1. 先计算我们需要从那个时间点开始抓 (Current Time - Target Duration)
    timeframe_map = {'1h': 3600, '4h': 14400, '1d': 86400}
    duration_sec = timeframe_map.get(timeframe, 3600)
    
    # 稍微多算一点时间缓冲 (1.1倍)，防止算少了
    span_ms = int(limit * duration_sec * 1000 * 1.1) 
    current_time = exchange.milliseconds()
    since = current_time - span_ms
    
    # 2. 循环抓取 (Retry Loop)
    retry_count = 0
    while len(all_ohlcv) < limit:
        try:
            # Kraken 每次最多给 720，我们显式请求 720
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=720)
            
            if not batch:
                if retry_count < 3: # 如果没抓到，重试3次
                    retry_count += 1
                    time.sleep(1)
                    continue
                else:
                    break # 真的没了
            
            # 重置重试计数
            retry_count = 0
            
            # 更新时间指针：从刚才抓到的最后一条数据的“下一秒”开始
            last_timestamp = batch[-1][0]
            
            # 防止死循环：如果新抓到的数据和上次一样，强制跳过
            if len(all_ohlcv) > 0 and last_timestamp <= all_ohlcv[-1][0]:
                since += duration_sec * 1000 * 10 # 强制往后跳10根K线
                continue

            # 拼接到总数据里
            all_ohlcv += batch
            since = last_timestamp + 1000 # 往后推1秒
            
            # 更新进度条
            percent = min(len(all_ohlcv) / limit, 1.0)
            my_bar.progress(percent, text=f"{progress_text} ({len(all_ohlcv)}/{limit})")
            
            # 只要抓到了接近现在的最新数据，就停止
            if last_timestamp >= current_time - (duration_sec * 1000):
                break
                
            # 休息一下，防止被封IP
            time.sleep(0.5)
            
        except Exception as e:
            st.warning(f"下载小插曲: {e}，正在重试...")
            time.sleep(1)
            retry_count += 1
            if retry_count > 5: break
    
    my_bar.empty()
    
    # 如果抓少了，给出提示但继续运行
    if len(all_ohlcv) < limit:
        st.warning(f"⚠️ 交易所限制，实际获取到 {len(all_ohlcv)} 条数据 (已足够分析)。")
    else:
        # 截取掉多余的，只留最新的 limit 条
        all_ohlcv = all_ohlcv[-limit:]

    # 转 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
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

# ---------------------------------------------------------
# 4. 训练与预测
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
# 5. 主界面
# ---------------------------------------------------------
st.sidebar.header("控制面板")
symbol = st.sidebar.text_input("交易对", "DOGE/USD") # 必须是大写 USD
timeframe = st.sidebar.selectbox("周期", ['1h', '4h', '1d'])
limit_num = st.sidebar.slider("学习 K 线数量", 500, 5000, 3000, step=100)

if st.button("开始分析", type="primary"):
    try:
        # 获取数据
        df = fetch_and_prepare_data(symbol, timeframe, limit=limit_num)
        
        st.toast(f"成功获取 {len(df)} 条数据，AI 正在思考...", icon="🧠")
        
        model, acc, pred, prob = train_and_predict(df)
        
        # 结果计算
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
        c1, c2, c3 = st.columns(3)
        c1.metric("当前价格", f"${last_close:.5f}")
        c2.metric("AI 建议", direction)
        c3.metric("信心指数", f"{prob[pred]*100:.1f}%")
        
        st.info(f"📊 **基于 {len(df)} 根 K 线 ({timeframe}) 的策略建议**")
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

    except Exception as e:
        st.error(f"发生错误: {e}")
