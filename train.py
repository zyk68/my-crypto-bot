import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. 页面基础配置
# ---------------------------------------------------------
st.set_page_config(page_title="AI 机器学习预测", layout="wide")
st.title("🧠 机器学习实战：ETH 价格预测")
st.caption("使用随机森林 (Random Forest) 算法，根据历史数据自动学习交易规律")

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
# 3. 数据处理与特征工程 (教 AI 认识数据)
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe):
    st.toast(f"正在下载 {symbol} 的历史数据并进行教学...", icon="🎓")
    
    # 1. 获取数据 (限制500条以防内存溢出)
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=500)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 2. 构造特征 (Feature Engineering) - 这是 AI 学习的教材
    # 简单特征：RSI, 均线差, 价格变化率, 波动率
    df['returns'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # RSI 计算
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 均线偏离度
    df['SMA_7'] = df['close'].rolling(7).mean()
    df['dist_SMA_7'] = (df['close'] - df['SMA_7']) / df['SMA_7']
    
    # 3. 构造目标 (Label) - 告诉 AI 什么是“正确答案”
    # 如果下一根K线收盘价 > 当前收盘价，标记为 1 (涨)，否则为 0 (跌)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 清除空值 (刚开始计算指标时会有NaN)
    df.dropna(inplace=True)
    return df

# ---------------------------------------------------------
# 4. 训练模型与预测
# ---------------------------------------------------------
def train_and_predict(df):
    # 定义 AI 要看哪些指标 (Features)
    feature_cols = ['RSI', 'dist_SMA_7', 'returns', 'range', 'volume']
    
    X = df[feature_cols] # 试卷题目
    y = df['Target']     # 标准答案
    
    # 切分数据：80%用来学习，20%用来模拟考试
    # shuffle=False 非常重要，因为时间序列不能打乱
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # 初始化模型：随机森林
    model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
    
    # 开始训练 (Fit)
    model.fit(X_train, y_train)
    
    # 模拟考试
    test_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)
    
    # 实战预测：用最新一行数据预测未来
    latest_features = X.iloc[[-1]] # 取最后一行
    future_pred = model.predict(latest_features)[0]
    future_prob = model.predict_proba(latest_features)[0] # 获取概率
    
    return model, accuracy, future_pred, future_prob, feature_cols

# ---------------------------------------------------------
# 5. 主界面逻辑
# ---------------------------------------------------------
st.sidebar.header("控制面板")
symbol = st.sidebar.text_input("交易对", "ETH/USD")
timeframe = st.sidebar.selectbox("周期", ['1h', '4h', '1d'])

if st.button("开始 AI 训练与预测", type="primary"):
    with st.spinner("AI 正在疯狂计算中..."):
        try:
            # 1. 获取数据
            df = fetch_and_prepare_data(symbol, timeframe)
            
            # 2. 训练模型
            model, acc, pred, prob, feat_cols = train_and_predict(df)
            
            # --- 结果展示区 ---
            
            # 顶部关键指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("当前价格", f"${df['close'].iloc[-1]:.2f}")
            col2.metric("AI 历史准确率", f"{acc*100:.1f}%", help="模型在最近20%未见过的历史数据上的预测胜率")
            
            # 预测结果
            direction = "看涨 (UP) 📈" if pred == 1 else "看跌 (DOWN) 📉"
            confidence = prob[pred] * 100
            
            # 根据置信度变色
            color = "normal"
            if confidence > 60: color = "inverse"
            
            with col3:
                st.metric("AI 预测下个周期", direction)
            with col4:
                st.metric("AI 信心指数", f"{confidence:.1f}%")

            st.divider()

            # 图表区
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader("📊 价格走势与均线")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='K线'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_7'], line=dict(color='orange'), name='SMA 7'))
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("🧠 AI 觉得什么最重要？")
                # 显示特征重要性 (Feature Importance)
                importances = pd.DataFrame({
                    '特征': feat_cols,
                    '重要性': model.feature_importances_
                }).sort_values('重要性', ascending=True)
                
                fig_imp = go.Figure(go.Bar(
                    x=importances['重要性'],
                    y=importances['特征'],
                    orientation='h'
                ))
                fig_imp.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_imp, use_container_width=True)

            with st.expander("查看 AI 学习用的原始数据"):
                st.dataframe(df.tail(10))

        except Exception as e:
            st.error(f"运行出错: {e}")
