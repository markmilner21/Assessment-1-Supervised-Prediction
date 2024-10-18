import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

indep_train = train_data[['Stringency Index', 'CH Index', 'Gov Resp Index', 'Econ Sup Index', 'days_since']]
dep_train = train_data['reproduction_rate']
indep_test = test_data[['Stringency Index', 'CH Index', 'Gov Resp Index', 'Econ Sup Index', 'days_since']]
dep_test = train_data['reproduction_rate']

# We will deal with missing values initially by simply removing rows containing missing values

train_data_cleaned = train_data.dropna(subset=['Stringency Index', 'CH Index', 'Gov Resp Index', 'Econ Sup Index', 'days_since', 'reproduction_rate'])
test_data_cleaned = test_data.dropna(subset=['Stringency Index', 'CH Index', 'Gov Resp Index', 'Econ Sup Index', 'days_since', 'reproduction_rate'])

indep_train = train_data_cleaned[['Stringency Index', 'CH Index', 'Gov Resp Index', 'Econ Sup Index', 'days_since']]
dep_train = train_data_cleaned['reproduction_rate']
indep_test = test_data_cleaned[['Stringency Index', 'CH Index', 'Gov Resp Index', 'Econ Sup Index', 'days_since']]
dep_test = test_data_cleaned['reproduction_rate']

# Knn models are distance based, so scaling of features (indep. variables) is essential

scaler = StandardScaler()

# We apply the function fit_transform to the training data which chooses model parameters according to the mean and standard deviation
# We apply simply the transform function to the test data since the model parameters have already been chosen

indep_scaled_train = scaler.fit_transform(indep_train)
indep_scaled_test = scaler.transform(indep_test)

# First we will use k=5

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(indep_scaled_train, dep_train)
prediction = knn.predict(indep_scaled_test)

# Now we will test the performance of our model

mse = mean_squared_error(dep_test, prediction)
print(f'Mean Squared Error: {mse}')


