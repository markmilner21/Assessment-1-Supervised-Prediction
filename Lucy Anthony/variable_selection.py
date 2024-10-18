# Now we will consider the hyperparameters selected through parameter optimisation (p=1 (Manhattan Metric), k=26, uniform weights)
# To further improve our model we will experiment with feature selection using recursive feature elimination (RFE)
# Since knn does not inherently provide a way to rank the importance of features, we will also include a linear model to aid with varibale selection

import Knn5
from sklearn.feature_selection import RFE
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# We will also implement our chosen parameters here.

X_train = Knn5.indep_train
X_test = Knn5.indep_test
Y_train = Knn5.dep_train
Y_test = Knn5.dep_test

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

base_model = LinearRegression()
knn_model = KNeighborsRegressor(n_neighbors=26, p=1, weights = 'uniform')

rfe = RFE(estimator=base_model, n_features_to_select=3)
rfe.fit(X_train, Y_train)
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Now we use the variables selected using RFE and the linear regression model and we test the performance of the optimised knn model using these variables

knn_model.fit(X_train_rfe, Y_train)
predictions = knn_model.predict(X_test_rfe)
mse = mean_squared_error(Y_test, predictions)

print(f'Mean Squared Error for k=26, p=1, weights="uniform", with selected features: {mse}')
print(f'Selected Features: {rfe.support_}')

# Although the mse was actually worse here, we can see that the Recursive Feature Elimination selected the features: 'CH Index', 'Gov Resp Index', 'Econ Sup Index' as statistically significant
# Thus we can try performing 3-fold validation with just these 3 variables

scores = cross_val_score(knn_model, X_train, Y_train, cv=3, scoring='neg_mean_squared_error')
average_mse = -scores.mean()
print(f'Average MSE from 3-fold cross-validation for k=26, p=1, weights="uniform", with selected features: {average_mse}')

# Interestingly, using variable selection with the cross variable actually gives a very slightly higher mse than the model without variable selection