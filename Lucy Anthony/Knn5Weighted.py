import Knn5
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# By default, Knn assigns equal weight to all neighbors. However, giving more weight to closer neighbors, could improve model performance.

indep_scaled_train = Knn5.indep_scaled_train
dep_train = Knn5.dep_train
indep_scaled_test = Knn5.indep_scaled_test
dep_test = Knn5.dep_test

knn = KNeighborsRegressor(n_neighbors=5, p=1, weights='distance') 
knn.fit(indep_scaled_train, dep_train)
prediction = knn.predict(indep_scaled_test)

mse = mean_squared_error(dep_test, prediction)
print(f'Mean Squared Error for k=5 and Manhattan metric, weighted: {mse}')

plt.scatter(Knn5.dep_test, Knn5.prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("KNN Regression: Actual vs Predicted for k=5, Manhattan metric, weighted")
plt.show()

# Slightly worse mse, scatter plot roughly the same