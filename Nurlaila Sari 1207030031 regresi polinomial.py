from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

#Database
# x = Data, y = Target
x = [[1], [3], [5], [7], [9], [11], [13], [15], [17], [19]]
y = [1, 9, 25, 49, 81, 121, 169, 225, 289, 361]

#Data uji
predict = np.array([[15]])
poly = PolynomialFeatures (degree=2)
x_ = poly.fit_transform(x)
predict_ = poly.fit_transform(predict)
regr = linear_model.LinearRegression()
regr.fit(x_,y)

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict (predict_))
