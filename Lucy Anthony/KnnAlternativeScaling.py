import Knn5
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

dep_train = Knn5.dep_train
indep_train = Knn5.indep_train
dep_test = Knn5.dep_test
indep_test = Knn5.indep_test

scaler = MinMaxScaler()
indep_scaled_train = scaler.fit_transform(indep_train)
indep_scaled_test = scaler.transform(indep_test)

knn = KNeighborsRegressor(n_neighbors=8, p=1)
knn.fit(indep_scaled_train, dep_train)
prediction = knn.predict(indep_scaled_test)

mse = mean_squared_error(dep_test, prediction)
print(f'Mean Squared Error for k=8 and Manhattan metric, MinMaxScaler: {mse}')

# MinMaxScaler is somewhat better, but not dramatically better. Now we will try RobustScaler.

scaler = RobustScaler()

indep_scaled_train = scaler.fit_transform(indep_train)
indep_scaled_test = scaler.transform(indep_test)

knn = KNeighborsRegressor(n_neighbors=8, p=1)
knn.fit(indep_scaled_train, dep_train)
prediction = knn.predict(indep_scaled_test)

mse = mean_squared_error(dep_test, prediction)
print(f'Mean Squared Error for k=8 and Manhattan metric, RobustScaler: {mse}')

# RobustScaler is even better (WHY?)
# From now on, we will use the hyperparameters of RobustScaler, Manhattan metric and a higher value of k (investigate optimal value later)