import Knn5
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# One hyperparemter that we can experiment with changing is the metric.
# Knn uses distance metrics to determine neighbors, and the default distance metric is Euclidean distance. However we can try with Manhattan distance.

indep_scaled_train = Knn5.indep_scaled_train
dep_train = Knn5.dep_train
indep_scaled_test = Knn5.indep_scaled_test
dep_test = Knn5.dep_test

knn = KNeighborsRegressor(n_neighbors=5, p=1) 
knn.fit(indep_scaled_train, dep_train)
prediction = knn.predict(indep_scaled_test)

mse = mean_squared_error(dep_test, prediction)
print(f'Mean Squared Error for k=5 and Manhattan metric: {mse}')

plt.scatter(Knn5.dep_test, Knn5.prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("KNN Regression: Actual vs Predicted for k=5, Manhattan metric")
plt.show()

# The Manhattan plot has slightly lower MSE (why?) but is not much better overall, as shown by the scatter graph.