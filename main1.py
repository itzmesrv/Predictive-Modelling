import pandas as pd
import pickle
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_valid_origins_from_csv(csvfile): # valid origins
    df = pd.read_csv(csvfile)
    origin_unique = df[['origin_city', 'origin_state']].drop_duplicates().sort_values(by='origin_city')
    valid_origin = set(tuple(x) for x in origin_unique.values)
    return valid_origin
    
def load_valid_destinations_from_csv(csvfile): # valid destinations
    df = pd.read_csv(csvfile)
    destination_unique = df[['destination_city', 'destination_state']].drop_duplicates().sort_values(by='destination_city')
    valid_destination = set(tuple(x) for x in destination_unique.values)
    return valid_destination

def save_data_as_pickle(data, filename): # to save file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def train_and_save_model(filepath, model_filename="trained_model.pkl", encoder_filename = "encoder.pkl", test_data_filename="test_data.pkl",valid_origin_filename="valid_origin.pkl",valid_destination_filename="valid_destination.pkl"):
    # Preprocessing
    df = pd.read_csv(filepath)
    df = df.drop(columns=['id', 'origin_zip', 'destination_zip', 'fuel_surcharge', 'created_userid', 'cache_date', 'updated_at', 'updated_userid', 'meta_data', 'rate_source', 'duration', 'max_rate', 'avg_rate', 'miles'])
    df = df.drop_duplicates(subset=['origin_city', 'origin_state', 'destination_city', 'destination_state','created_at'])

    # Handling date column
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['day'] = df['created_at'].dt.day
    df['month'] = df['created_at'].dt.month
    df['year'] = df['created_at'].dt.year
    df = df.drop(columns=['created_at'])

    # One-hot encoding
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore') # One hot encoder
    encoded_cities = encoder.fit_transform(df[['equipment_type','origin_city','origin_state','destination_city','destination_state','day','month','year']]) # encoding
    encoded_cities_df = pd.DataFrame(encoded_cities, columns=encoder.get_feature_names_out(['equipment_type','origin_city','origin_state','destination_city','destination_state','day','month','year'])) #df
    df.reset_index(drop=True, inplace=True) # resetting index of original df
    encoded_cities_df.reset_index(drop=True, inplace=True) # resetting index of encoded df
    new_df = pd.concat([df, encoded_cities_df], axis=1).drop(columns=['equipment_type','origin_city','origin_state','destination_city','destination_state','day','month','year']) # concat 

    # Splitting data
    X = new_df.drop(columns=['target_rate'])
    y = new_df['target_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = { # best parameters from Grid Search
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 300
    }

    model = RandomForestRegressor( #  model with best params
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        n_estimators=best_params['n_estimators'],
        random_state=42
    )

    model.fit(X_train, y_train)
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model trained and saved as {model_filename}") # saving model

    with open(encoder_filename, 'wb') as file:
        pickle.dump(encoder, file)
    print(f"Encoder saved as {encoder_filename}") # saving encoder

    with open(test_data_filename, 'wb') as file:
        pickle.dump((X_test, y_test), file)
    print(f"Test data saved as {test_data_filename}") # saving test data

if __name__ == "__main__":
    filename = input("CSV file: ")
    filepath = os.path.join(os.getcwd(), filename)

    valid_origin = load_valid_origins_from_csv(filepath)
    valid_destination = load_valid_destinations_from_csv(filepath)
    save_data_as_pickle(valid_origin, 'valid_origin.pkl')
    print(f"Valid origins saved as valid_origin.pkl")
    save_data_as_pickle(valid_destination, 'valid_destination.pkl')
    print(f"Valid destinations saved as valid_destination.pkl")

    train_and_save_model(filepath)