import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def randomForest_train_predict(x_train, y_train, x_test):

    # Create model
    model = RandomForestRegressor()
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'max_features': ['log2', 'sqrt'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['mse', 'squared_error']
    }


    model.fit(x_train, y_train)

    # Predictions
    predictions = model.predict(x_test)

    return predictions
