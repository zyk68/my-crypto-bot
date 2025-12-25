import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# 1. 页面配置
# ---------------------------------------------------------
st.set_page_config(page_title="AI 交易机器人", layout="wide")
st.title("🤖 AI 币安量化交易监控")

# ---------------------------------------------------------
# 2. 连接币安交易所 (使用公共接口，无需 API Key 也能看行情)
# ---------------------------------------------------------
@st.cache_resource
def init_exchange():
    return ccxt.kraken({
        'enableRateLimit': True,  # 启用速率限制，防止报错
        # 'apiKey': st.secrets["BINANCE_API_KEY"], # 如果你要交易，后期在 Secrets 里填入 Key
        # 'secret': st.secrets["BINANCE_SECRET_KEY"],
    })

try:
    exchange = init_exchange()
    st.success(f"✅ 成功连接到交易所: {exchange.name}")
except Exception as e:
    st.error(f"无法连接交易所: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 获取数据函数
# ---------------------------------------------------------
def fetch_data(symbol='ETH/USDT', timeframe='1h', limit=100):
    st.write(f"正在获取 {symbol} 的 {timeframe} 数据...")
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ---------------------------------------------------------
# 4. 主运行逻辑
# ---------------------------------------------------------
# 侧边栏控制
st.sidebar.header("参数设置")
symbol = st.sidebar.text_input("交易对", "ETH/USDT")
timeframe = st.sidebar.selectbox("时间周期", ['15m', '1h', '4h', '1d'])

# 按钮触发
if st.button("开始分析 / 刷新数据"):
    try:
        # 获取数据
        df = fetch_data(symbol, timeframe)
        
        # 显示最新价格
        last_price = df['close'].iloc[-1]
        st.metric(label=f"{symbol} 最新价格", value=f"${last_price:.2f}")

        # 画 K 线图
        fig = go.Figure(data=[go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])
        fig.update_layout(title=f"{symbol} K线图", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # 显示原始数据表
        with st.expander("查看详细数据"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"发生错误: {e}")

# ---------------------------------------------------------
# 5. 这里可以继续写你的 AI 训练逻辑
# ---------------------------------------------------------
# import sklearn...
# model = ...
