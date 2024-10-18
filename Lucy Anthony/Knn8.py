import Knn5
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

indep_scaled_train = Knn5.indep_scaled_train
dep_train = Knn5.dep_train
indep_scaled_test = Knn5.indep_scaled_test
dep_test = Knn5.dep_test

knn = KNeighborsRegressor(n_neighbors=8)
knn.fit(indep_scaled_train, dep_train)
prediction = knn.predict(indep_scaled_test)

mse = mean_squared_error(dep_test, prediction)
print(f'Mean Squared Error for k=8: {mse}')