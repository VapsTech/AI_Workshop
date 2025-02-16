import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

def svr_train_predict(x_train, y_train, x_test):
    # Scale the data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # Scale the data
    X_train_scaled = scaler_X.fit_transform(x_train)
    X_test_scaled = scaler_X.transform(x_test)

    Y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1, 1))

    # Create the Machine Learning model
    model = SVR()

    # Define the hyperparameters to tune
    param_grid = {
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'C': [1, 10],
        'gamma': ['scale']
    }

    # GridSearchCV to get the best model
    gridSearch = GridSearchCV(estimator= model, param_grid= param_grid, cv= 3, n_jobs= -1, verbose= 2)

    # Fit the model
    gridSearch.fit(X_train_scaled, Y_train_scaled.ravel())

    model = gridSearch.best_estimator_  # return the BEST model

    # Predictions
    predictions_scaled = model.predict(X_test_scaled)
    # Inverse the scaling to Orignal Values
    predictions = scaler_Y.inverse_transform(predictions_scaled.reshape(-1, 1))

    return predictions