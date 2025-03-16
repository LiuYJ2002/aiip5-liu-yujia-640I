
import json
from load_data import fetch_data
from preprocess import preprocess_data, split_data
from model import train_temperature_model, train_plant_stage_model

def load_config(config_file):
    """
    Load configuration from JSON file
    """
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    # Load configuration
    config = load_config('config.json')
    
    # Fetch data
    df = fetch_data('src/data/agri.db')
    
    # Preprocess data
    df = preprocess_data(df, config['data'])
    
    # Split data for temperature prediction
    X_train, X_test, y_train, y_test = split_data(df, target_column='Temperature Sensor (°C)', type = 'regression')
    train_temperature_model(X_train, X_test, y_train, y_test, config['temperature_model'])
    
    # Split data for plant stage classification
    X_train, X_test, y_train, y_test = split_data(df, target_column='Plant Type-Stage', type = 'classification')
    train_plant_stage_model(X_train, X_test, y_train, y_test, config['plant_stage_model'])

if __name__ == "__main__":
    main()
