import KnnAlternativeScaling
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

knn = KnnAlternativeScaling.knn
indep_scaled_train = KnnAlternativeScaling.indep_scaled_train
dep_train = KnnAlternativeScaling.dep_train

scores = cross_val_score(knn, indep_scaled_train, dep_train, cv=3, scoring='neg_mean_squared_error')
average_mse = -scores.mean()
print(f"Average MSE from 3-fold cross-validation: {average_mse}")

# Note the average mse is actually worse than from the previous model without cross-validation. This could be caused by skewed data (non-uniform), or too many folds in the cross-validation.
# Note that 3-fold cross-validation performs better than 5-fold validation, but still worse than the previous model.
# Another thing to try is further optimisation of the hyperparameters using GridSearchCV.

param_grid1 = {
    'n_neighbors' : [8,12, 16, 20, 24,28], 'weights' : ['uniform', 'distance'], 'p' : [1,2]
}

grid_search = GridSearchCV(KNeighborsRegressor(), param_grid1, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(indep_scaled_train, dep_train)

print("Best Parameters 3-fold validation: ", grid_search.best_params_)
print("Best MSE 3-fold validation: ", -grid_search.best_score_)

# Optimising hyperparameters and using 3-fold cross validation gives us mse of 0.1958 using k=26, p=1 (Manhattan Distance), and uniform weights.
# We will also check for 5-fold validation to see if this is better or worse

param_grid2 = {
    'n_neighbors' : [8,12, 16, 20, 24,28], 'weights' : ['uniform', 'distance'], 'p' : [1,2]
}

grid_search = GridSearchCV(KNeighborsRegressor(), param_grid2, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(indep_scaled_train, dep_train)

print("Best Parameters 5-fold validation: ", grid_search.best_params_)
print("Best MSE 5-fold validation: ", -grid_search.best_score_)

# 5-fold validation gives a slightly worse mse so we will stick with 3-fold validation.