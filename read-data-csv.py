from time import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
# 导入文件
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

os.chdir(r'data-csv')  # 这里是图片目录
file_list = [x for x in os.listdir() if x.endswith(".csv")]
print(file_list)
x_label = ['x_location', 'y_location',
           940.70, 947.50, 954.20, 961.00, 967.70, 974.50, 981.20, 988.00, 994.70,
           1001.50, 1008.20, 1015.00, 1021.70, 1028.50, 1035.20, 1042.00,
           1048.70, 1055.50, 1062.20, 1069.00, 1075.70, 1082.50, 1089.20, 1096.00,
           1102.70, 1109.50, 1116.20, 1123.00, 1129.70, 1136.50, 1143.20,
           1150.00, 1156.70, 1163.50, 1170.20, 1177.00, 1183.70, 1190.50, 1197.20,
           1204.00, 1210.70, 1217.50, 1224.20, 1231.00, 1237.70, 1244.50,
           1251.20, 1258.00, 1264.70, 1271.50, 1278.20, 1285.00, 1291.70, 1298.50,
           1305.20, 1312.00, 1318.70, 1325.50, 1332.20, 1339.00, 1345.70,
           1352.50, 1359.20, 1366.00, 1372.70, 1379.50, 1386.20, 1393.00, 1399.70,
           1406.50, 1413.20, 1420.00, 1426.80, 1433.50, 1440.30, 1447.00,
           1453.80, 1460.50, 1467.30, 1474.00, 1480.80, 1487.50, 1494.30, 1501.00,
           1507.80, 1514.50, 1521.30, 1528.00, 1534.80, 1541.50, 1548.30,
           1555.00, 1561.80, 1568.50, 1575.30, 1582.00, 1588.80, 1595.50, 1602.30,
           1609.00, 1615.80, 1622.50, 1629.30, 1636.00, 1642.80, 1649.50,
           1656.30, 1663.00, 1669.80, 1676.50, 1683.30, 1690.00, 1696.80, 1703.50,
           1710.30, 1717.00, 1723.80, 1730.50, 1737.30, 1744.00, 1750.80,
           1757.50, 1764.30, 1771.00, 1777.80, 1784.50, 1791.30, 1798.00, 1804.80,
           1811.50, 1818.30, 1825.00, 1831.80, 1838.50, 1845.30, 1852.00,
           1858.80, 1865.50, 1872.30, 1879.00, 1885.80, 1892.50, 1899.30, 1906.00,
           1912.80, 1919.50, 1926.30, 1933.00, 1939.80, 1946.50, 1953.30,
           1960.00, 1966.80, 1973.50, 1980.30, 1987.00, 1993.80, 2000.50, 2007.30,
           2014.00, 2020.80, 2027.50, 2034.30, 2041.00, 2047.80, 2054.50,
           2061.30, 2068.00, 2074.80, 2081.60, 2088.30, 2095.10, 2101.80, 2108.60,
           2115.30, 2122.10, 2128.80, 2135.60, 2142.30, 2149.10, 2155.80,
           2162.60, 2169.30, 2176.10, 2182.80, 2189.60, 2196.30, 2203.10, 2209.80,
           2216.60, 2223.30, 2230.10, 2236.80, 2243.60, 2250.30, 2257.10,
           2263.80, 2270.60, 2277.30, 2284.10, 2290.80, 2297.60, 2304.30, 2311.10,
           2317.80, 2324.60, 2331.30, 2338.10, 2344.80, 2351.60, 2358.30,
           2365.10, 2371.80, 2378.60, 2385.30, 2392.10, 2398.80, 2405.60, 2412.30,
           2419.10, 2425.80, 2432.60, 2439.30, 2446.10, 2452.80, 2459.60,
           2466.30, 2473.10, 2479.80, 2486.60, 2493.30, 2500.10, 2506.80, 2513.60,
           2520.30, 2527.10, 2533.80, 2540.60, 2547.30, 2554.10, 2560.80,
           2567.60, 2574.30, 2581.10, 2587.80, 2594.60, 2601.30, 2608.10, 2614.80,
           2621.60, 2628.30, 2635.10, 2641.80, 2648.60, 2655.30, 2662.10]

total_data_label = x_label[2:258]
total_data_label.insert(0, '序号')
total_data = pd.DataFrame(columns=total_data_label)
file_order = 0
new_file_list = []
for file in file_list:
    print("----第{}个文件开始----".format(file_order + 1))
    data = pd.read_csv(file, header=8, names=x_label)
    # 除去坐标列
    data_new = data.drop(['x_location', 'y_location'], axis=1)
    # 除去异常数据
    if len(data_new[(data_new[1028.50] > 1)]) > 0 or len(data_new[(data_new[2250.30] > 1)]) > 1:
        print("----第{}个文件数据有误----跳过".format(file_order + 1))
        continue
    # 绘制该图样的光谱反射率的折线图
    mean_data = np.mean(data_new)
    plt.plot(x_label[2:258], mean_data.values)    # 全部光谱是2:258
    new_file_list.append(file)
    # 将处理后的土样数据添加到新表中做处理
    total_data.loc[file_order] = mean_data
    total_data.loc[file_order, '序号'] = file.split('.')[0]
    print("第{}个土样计算均值并添加成功".format(file_order + 1))

    file_order = file_order + 1

# 读取结果数据
result = pd.read_excel('result.xlsx')
merge_data = pd.merge(total_data, result, on='序号', how='left')
print(merge_data)

plt.legend(new_file_list)
plt.ylim(0, 1)
plt.show()

# 取X，Y
X = merge_data[x_label[2:258]].values

y = merge_data['SOC g/kg'].values
print(X)
print(y)

# 划分训练集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=20, shuffle=True)
print("X_train")
print(X_train.shape)
print(X_train)
print("y_train")
print(y_train.shape)
print(y_train)

# # 数据预处理
# Stand_X = StandardScaler()
# Stand_Y = StandardScaler()
# X_train = Stand_X.fit_transform(X_train)
# X_val = Stand_X.fit_transform(X_val)
# y_train = Stand_Y.fit_transform(y_train.reshape(-1, 1))
# y_val = Stand_Y.fit_transform(y_val.reshape(-1, 1))

print("X_val")
print(X_val)
print("y_val")
print(y_val)

print("--------------------------开始预测--------------------------")
svr = SVR(kernel='poly', C=10, degree=2)
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_val)
print("SVR：")
print("result：", svr_pred)
print("训练集 R：", svr.score(X_train, y_train))
print("测试集 R：", svr.score(X_val, y_val))
print("----------------------------------------------------------")
plsr = PLSRegression(n_components=2)
plsr.fit(X_train, y_train)
plsr_pred = plsr.predict(X_val)
print("PLSR ：")
print("result：", plsr_pred)
print("训练集 R：", plsr.score(X_train, y_train))
print("测试集 R：", plsr.score(X_val, y_val))
