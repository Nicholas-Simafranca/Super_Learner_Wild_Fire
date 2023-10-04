from math import sqrt
from numpy import hstack
from numpy import vstack
import pandas as pd
from numpy import asarray
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor


# define neural network model
class Regressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 64)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 32)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(16, 8)
        self.act4 = nn.ReLU()
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.output(x)
        return x

# sklearn-compatible wrapper for the PyTorch model
class TorchNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_epochs=200, batch_size=115, learning_rate=0.01, n_features=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = Regressor(n_features)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        for epoch in range(self.n_epochs):
            for i in range(0, len(X), self.batch_size):
                Xbatch = X[i:i + self.batch_size]
                ybatch = y[i:i + self.batch_size]
                y_pred = self.model(Xbatch)
                loss = self.loss_fn(y_pred.squeeze(), ybatch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        y_pred = self.model(X)
        return y_pred.detach().numpy().squeeze()

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# create a list of base-models
def get_models(n_features):
    models = list()
    models.append(ElasticNet())
    models.append(DecisionTreeRegressor())
    models.append(Ridge(alpha=1))
    models.append(Lasso(alpha=1))
    models.append(KNeighborsRegressor(n_neighbors=6, weights="distance"))
    models.append(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42))
    models.append(XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=20, random_state=42))
    models.append(BaggingRegressor(n_estimators=100))
    models.append(RandomForestRegressor(n_estimators=100))
    models.append(ExtraTreesRegressor(n_estimators=100))
    models.append(TorchNNRegressor(n_features=n_features))
    return models

def get_out_of_fold_predictions(X, y, models, latitude, longitude):
    meta_X, meta_y, meta_longitude, meta_latitude = list(), list(), list(), list()
    # define split of data
    kfold = KFold(n_splits=10, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
        # get data
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        test_longitude = longitude[test_ix]
        test_latitude = latitude[test_ix]
        meta_y.extend(test_y)
        meta_longitude.extend(test_longitude)
        meta_latitude.extend(test_latitude)
        # fit and make predictions with each sub-model
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            # store columns
            fold_yhats.append(yhat.reshape(len(yhat),1))
        # store fold yhats as columns
        meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y), asarray(meta_latitude), asarray(meta_longitude)


# fit all base models on the training dataset
def fit_base_models(X, y, models):
    predictions = {}
    for model in models:
        model.fit(X, y)
        yhat = model.predict(X)
        predictions[model.__class__.__name__] = yhat
    return predictions


# fit a meta model
def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
        mse = mean_squared_error(y, yhat)
        print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))

def super_learner_predictions(X, y, latitude, longitude, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict(X)
        meta_X.append(yhat.reshape(len(yhat),1))
    meta_X = hstack(meta_X)
    # Create DataFrame from predicted values
    df_meta_X = pd.DataFrame(meta_X)
    # Add y, latitude and longitude columns to the DataFrame
    df_meta_X['dnbr'] = y
    df_meta_X['Lat'] = latitude
    df_meta_X['Lon'] = longitude
    # Save df_meta_X to a csv file
    df_meta_X.to_csv("CZU_NEW_predicted_fitted_values.csv", index=False)
    # Predict
    return meta_model.predict(meta_X)



data = pd.read_csv('prefire_czu_grouped_standard_finaldata.csv')

# Extract latitude and longitude
LC = data.iloc[:, 0]
lat = data.iloc[:, 1]
lon = data.iloc[:, 2]
y_pre = data.iloc[:, 4]

# Drop the unnecessary columns
features = data.drop(data.columns[[0, 1, 2, 3, 4]], axis=1)

# Combine standardized features, latitude, longitude, and y into a new DataFrame
df = pd.DataFrame(features, columns=features.columns)

# Create a new DataFrame with desired order of columns
df = pd.concat([LC, lat, lon, y_pre, df], axis=1)
df.columns = ['LC', 'Lat', 'Lon', 'dnbr'] + list(df.columns)[4:]


# Split the DataFrame into training and testing datasets
data, test_df = train_test_split(df, test_size=0.4, random_state=42)

# Save these training/testing datasets
data.to_csv("CZU_traindata.csv", index=False)
test_df.to_csv("CZU_testdata.csv", index=False)


latitude = data.iloc[:, 1].values
longitude = data.iloc[:, 2].values
y = data.iloc[:, 3].values  # Labels
X = data.drop(data.columns[[0, 1, 2, 3]], axis=1).values

lat_val = test_df.iloc[:, 1].values
lon_val = test_df.iloc[:, 2].values
test_df = test_df.drop(test_df.columns[[0, 1, 2]], axis=1)
y_val = test_df.iloc[:, 0].values
X_val = test_df.drop(test_df.columns[0], axis=1).values


# get models
models = get_models(n_features=X.shape[1])
# get out of fold predictions
meta_X, meta_y, meta_latitude, meta_longitude = get_out_of_fold_predictions(X, y, models, latitude, longitude)
meta_data = pd.DataFrame(meta_X)
meta_data['dnbr'] = meta_y
meta_data['Lat'] = meta_latitude
meta_data['Lon'] = meta_longitude


# Save to a csv file
meta_data.to_csv("CZU_NEW_meta_data.csv", index=False)

print('Meta CZU Final', meta_X.shape, meta_y.shape)
# fit base models
# Get fitted values for each model on the entire dataset
fitted_values = fit_base_models(X, y, models)

# Create a DataFrame from the dictionary
df_fitted_values = pd.DataFrame(fitted_values)

# Add y (dnbr), latitude and longitude columns to the DataFrame
df_fitted_values['dnbr'] = y
df_fitted_values['Lat'] = latitude
df_fitted_values['Lon'] = longitude

# Save df_fitted_values to a csv file
df_fitted_values.to_csv("CZU_NEW_fitted_values.csv", index=False)


# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)

print('Coefficients: \n', meta_model.coef_)
# create a DataFrame from the coefficients
coefficients = pd.DataFrame(meta_model.coef_, columns=['Coefficient'])

# save the DataFrame to a csv file
coefficients.to_csv('CZU_NEW_coefficients.csv', index=False)

# evaluate base models
evaluate_models(X_val, y_val, models)
# evaluate meta model
yhat = super_learner_predictions(X_val, y_val, lat_val, lon_val, models, meta_model)


pd.DataFrame(yhat).to_csv("CZU_NEW_X_val_pred_super_learner.csv", index=False)
print('Super Learner: R^2 %.3f' % (r2_score(y_val, yhat)))
print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(y_val, yhat))))
