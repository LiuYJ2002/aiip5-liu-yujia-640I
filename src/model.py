from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

def create_model(model_config, X_train, y_train):
    """
    Create and optionally tune a machine learning model based on configuration.
    """
    model_type = model_config['type']
    params = model_config.get('params', {})
    tuning = model_config.get('tuning', None)
    params['verbose'] = 1  
    # Supported models
    model_classes = {
        'RandomForestRegressor': RandomForestRegressor,
        'RandomForestClassifier': RandomForestClassifier,
        'MLPRegressor': MLPRegressor,
        'MLPClassifier': MLPClassifier
    }

    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_classes[model_type](**params)

    # Perform hyperparameter tuning if enabled
    if tuning and 'search_space' in tuning:
        search_type = tuning.get('type', 'grid')  # Default to GridSearch
        search_space = tuning['search_space']
        cv = tuning.get('cv', 5)  

        if search_type == 'grid':
            search = GridSearchCV(model, search_space, cv=cv, n_jobs=-1)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, search_space, cv=cv, n_jobs=-1, n_iter=tuning.get('n_iter', 10))
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best parameters found for {model_type}: {search.best_params_}")

    return model


def train_temperature_model(X_train, X_test, y_train, y_test, model_config):
    """
    Train a model for predicting temperature with optional hyperparameter tuning.
    """
    model_type = model_config['type']
    model = create_model(model_config, X_train, y_train)
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Temperature Prediction MSE: {mse}")
    
    # Save the model
    if model_type == 'MLPRegressor':
        joblib.dump(model, 'temperature_modelNN.pkl')
    else:
        joblib.dump(model, 'temperature_model_tree.pkl')

def train_plant_stage_model(X_train, X_test, y_train, y_test, model_config):
    """
    Train a model for classifying plant stage with optional hyperparameter tuning.
    """
    model = create_model(model_config, X_train, y_train)
    model.fit(X_train, y_train)
    model_type = model_config['type']
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Plant Stage Classification Accuracy: {accuracy}")
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
    # Save the model
    if model_type == 'MLPClassifier':
        joblib.dump(model, 'plant_stage_modelNN.pkl')
    else:
        joblib.dump(model, 'plant_stage_model_tree.pkl')
