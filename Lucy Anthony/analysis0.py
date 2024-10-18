import Knn5
import Knn6
import Knn7
import Knn8
import matplotlib.pyplot as plt

# Here, we will graph our results and consider adjusting the hyperparameters in order to improve model performance
# First we will create a scatter plot of the actual vs predicted values from the 5-nn initial model

plt.scatter(Knn5.dep_test, Knn5.prediction)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("KNN Regression: Actual vs Predicted for k=5")
plt.show()

# This graph shows that, while points show a roughly positive correlation, they are concentrated around the lower true values, suggesting that the KNN model might be struggling to make accurate predictions, particularly when the true values are higher. 
# The points not lying close to the diagonal line imply prediction errors.
# The first improvement we are going to make is to try different values of k.

predictions_k5 = Knn5.prediction
predictions_k6 = Knn6.prediction
predictions_k7 = Knn7.prediction
predictions_k8 = Knn8.prediction

true_values = Knn5.dep_test

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot for k=5
axes[0, 0].scatter(true_values, predictions_k5)
axes[0, 0].set_title('KNN Regression: k=5')
axes[0, 0].set_xlabel('True Values')
axes[0, 0].set_ylabel('Predictions')

# Plot for k=6
axes[0, 1].scatter(true_values, predictions_k6)
axes[0, 1].set_title('KNN Regression: k=6')
axes[0, 1].set_xlabel('True Values')
axes[0, 1].set_ylabel('Predictions')

# Plot for k=7
axes[1, 0].scatter(true_values, predictions_k7)
axes[1, 0].set_title('KNN Regression: k=7')
axes[1, 0].set_xlabel('True Values')
axes[1, 0].set_ylabel('Predictions')

# Plot for k=8
axes[1, 1].scatter(true_values, predictions_k8)
axes[1, 1].set_title('KNN Regression: k=8')
axes[1, 1].set_xlabel('True Values')
axes[1, 1].set_ylabel('Predictions')

plt.tight_layout()
plt.show()