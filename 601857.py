# _*_ coding: utf-8 _*_
'''
__author__ = 'Yuanqing Mei'
__email__ = 'dg1533019@smail.nju.edu.cn'
__file__ = 601857.py
__time__ = 2/26/20 12:14 AM
__description__ = ''
'''

# _*_ coding: utf-8 _*_
'''
__author__ = 'Yuanqing Mei'
__email__ = 'dg1533019@smail.nju.edu.cn'
__file__ = 600489.py
__time__ = 2/24/20 5:27 PM
__description__ = ''
'''

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

pd.set_option("expand_frame_repr", False)       # 当列太多时不换行
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


ts.set_token('168b4b447dde788ee239fa8d5b93c34cde6609bedbae5e96e939ef21')  # 需要输入自己的token
pro = ts.pro_api()


# 导入000002.SZ前复权日线行情数据，保留收盘价列
df = ts.pro_bar(ts_code='601857.SH', adj='qfq', start_date='20080101', end_date='20200226')
df.sort_values('trade_date', inplace=True)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)
df = df[['close']]
print(df.head())


# 计算当前、未来1-day涨跌幅
df['1d_future_close'] = df['close'].shift(-1)
df['1d_close_future_pct'] = df['1d_future_close'].pct_change(1)
df['1d_close_pct'] = df['close'].pct_change(1)
df['ma5'] = df['close'].rolling(5).mean()
df['ma5_close_pct'] = df['ma5'].pct_change(1)
df.dropna(inplace=True)
feature_names = ['当前涨跌幅方向', 'ma5当前涨跌幅方向']


df.loc[df['1d_close_future_pct'] > 0, '未来1d涨跌幅方向'] = '上涨'
df.loc[df['1d_close_future_pct'] <= 0, '未来1d涨跌幅方向'] = '下跌'

df.loc[df['1d_close_pct'] > 0, '当前涨跌幅方向'] = 1    # 上涨记为1
df.loc[df['1d_close_pct'] <= 0, '当前涨跌幅方向'] = 0   # 下跌记为0

df.loc[df['ma5_close_pct'] > 0, 'ma5当前涨跌幅方向'] = 1
df.loc[df['ma5_close_pct'] <= 0, 'ma5当前涨跌幅方向'] = 0


feature_and_target_cols = ['未来1d涨跌幅方向'] + feature_names
df = df[feature_and_target_cols]
print(df.head())


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# 创建特征 X 和标签 y
y = df['未来1d涨跌幅方向'].values
X = df.drop('未来1d涨跌幅方向', axis=1).values


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 创建一个k为6的k-NN分类器
knn = KNeighborsClassifier(n_neighbors=6)

# 放入训练集数据进行学习
knn.fit(X_train, y_train)

# 在测试集数据上进行预测
new_prediction = knn.predict(X_test)
print("Prediction: {}".format(new_prediction))

# 测算模型的表现：预测对的个数 / 总个数
print(knn.score(X_test, y_test))

print("the format of X is ", repr(X_test))
x_predict = [[1, 0]]
# x_predict = np.array([0, 0])
print("601857.SH stock' price for 20200227 is", knn.predict(x_predict))
