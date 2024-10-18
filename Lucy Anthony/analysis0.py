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

# These graphs all show that adjusting the value of k from 5 to 8 does not have much affect on the accuracy of the models predictions
# Equally the mse does not change dramatically
# We will also create a residuals plot to visualise whether the model is under or over-predicting

residuals_k5 = true_values - predictions_k5
residuals_k6 = true_values - predictions_k6
residuals_k7 = true_values - predictions_k7
residuals_k8 = true_values - predictions_k8

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot for k=5
axes[0, 0].scatter(predictions_k5, residuals_k5, color='r')
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residual Plot for k=5')

# Plot for k=6
axes[0, 1].scatter(predictions_k6, residuals_k6, color='b')
axes[0, 1].axhline(y=0, color='b', linestyle='--')
axes[0, 1].set_xlabel('Predicted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot for k=6')

# Plot for k=7
axes[1, 0].scatter(predictions_k7, residuals_k7, color='g')
axes[1, 0].axhline(y=0, color='g', linestyle='--')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot for k=7')

# Plot for k=8
axes[1, 1].scatter(predictions_k8, residuals_k8, color='y')
axes[1, 1].axhline(y=0, color='y', linestyle='--')
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residual Plot for k=8')

plt.tight_layout()
plt.show()

# The negative slope in the residuals suggests that the model is systematically under-predicting the higher values and possibly over-predicting some lower values
# Equally the residuals fan out slightly as the predicted values increase. This phenomenon is called heteroscedasticity, where the error variance changes across the range of predicted values. In this case, it seems the model struggles more with larger predicted values.

