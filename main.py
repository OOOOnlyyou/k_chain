import pandas as pd
import numpy as np
from k_chain import KChain

# -------------------------------------------Sample--------------------------------------------------------
# 获取某股数据
data = pd.read_csv('./stock_data/000002.SZ.csv').loc[:, ['trade_date', 'open', 'high', 'low', 'close', 'vol']]
# 选择时间
data = data[np.logical_and(data['trade_date'] >= 20200101, data['trade_date'] < 20210101)]
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data.set_index(['trade_date'], inplace=True)
data.sort_index(inplace=True)

k_chain = KChain(data)
k_chain.main(showBi=True, showSeg=True, showPivot=True)
