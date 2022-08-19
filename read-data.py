from sklearn.cross_decomposition import PLSRegression
X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
plsr = PLSRegression(n_components=2)
plsr.fit(X, Y)

plsr_pred = plsr.predict(X)
print(plsr_pred)
print("result：", plsr_pred)
print("PLSR ：")
print("训练集 R：", plsr.score(X, Y))
