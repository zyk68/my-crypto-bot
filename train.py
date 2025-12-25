# ---------------------------------------------------------
# 新增：分页下载增强版数据函数
# ---------------------------------------------------------
def fetch_and_prepare_data(symbol, timeframe, limit=3000): # <--- 这里默认改成了3000
    st.toast(f"正在深度挖掘 {limit} 条历史数据...", icon="⛏️")
    
    # --- 分页下载逻辑开始 ---
    all_ohlcv = []
    # 1. 计算大概的开始时间 (当前时间 - K线数量 * 周期时长)
    # timeframe_to_seconds 是 ccxt 的内部转换工具，需要手动算一下
    # 简单映射：1h = 3600s, 4h = 14400s
    timeframe_map = {'1h': 3600, '4h': 14400, '1d': 86400}
    duration_seconds = timeframe_map.get(timeframe, 3600) 
    
    # 算出 limit 根 K 线总共是多少毫秒
    span_ms = limit * duration_seconds * 1000
    current_time = exchange.milliseconds()
    since = current_time - span_ms
    
    # 2. 循环抓取
    placeholder = st.empty() # 占位符，用来显示进度
    while len(all_ohlcv) < limit:
        try:
            # 每次从 since 开始往后抓
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            if not ohlcv:
                break # 抓不到数据了就停
            
            # 拿到最后一条的时间，作为下一次抓取的起点
            last_time = ohlcv[-1][0]
            if last_time == since:
                break # 防止死循环（如果交易所不给新数据）
                
            since = last_time + 1 # 往后推1毫秒，避免重复
            all_ohlcv += ohlcv
            
            # 显示进度
            placeholder.text(f"已下载: {len(all_ohlcv)} / {limit} 根K线...")
            
            # 稍微休息一下，防止被交易所封IP
            import time
            time.sleep(0.1)
            
            # 如果已经抓到了现在的最新数据，就停
            if last_time >= current_time - duration_seconds*1000:
                break
                
        except Exception as e:
            st.warning(f"下载中途遇到小波动: {e}")
            break
            
    placeholder.empty() # 清除进度提示
    # 截取最后需要的 limit 条（因为可能抓多了）
    all_ohlcv = all_ohlcv[-limit:]
    # --- 分页下载逻辑结束 ---

    # 转成 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # --- 下面是之前的特征工程逻辑 (保持不变) ---
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

    # ATR 计算
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Target
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    return df
