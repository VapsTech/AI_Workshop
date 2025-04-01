import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def randomForest_train_predict(x_train, y_train, x_test):

    # Create model
    model = RandomForestRegressor()
    
    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'max_features': ['log2', 'sqrt'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['mse', 'squared_error']
    }

    # GridSearchCV to get the best model
    gridSearch = GridSearchCV(estimator= model, param_grid= param_grid, cv= 3, n_jobs= -1, verbose= 2)

    # Fit the model
    gridSearch.fit(x_train, y_train)

    model = gridSearch.best_estimator_ # return the BEST model

    # Predictions
    predictions = model.predict(x_test)

    return predictions
