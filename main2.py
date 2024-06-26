import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import r2_score

with open('valid_origin.pkl', 'rb') as f:
    valid_origin = pickle.load(f)

with open('valid_destination.pkl', 'rb') as f:
    valid_destination = pickle.load(f)

def load_model_and_encoder(model_filename, encoder_filename, test_data_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    with open(encoder_filename, 'rb') as file:
        encoder = pickle.load(file)
    with open(test_data_filename, 'rb') as file:
        X_test, y_test = pickle.load(file)
    return model, encoder, X_test, y_test

def preprocess_input(equipment_type, origin_city, origin_state, destination_city, destination_state, day, month, year, encoder, feature_names):
    user_input = {
        'equipment_type': equipment_type,
        'origin_city': origin_city,
        'origin_state': origin_state,
        'destination_city': destination_city,
        'destination_state': destination_state,
        'day': day,
        'month': month,
        'year': year
    }

    input_df = pd.DataFrame([user_input])
    encoded_input = encoder.transform(input_df)
    encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(input_df.columns))

    final_input_df = pd.DataFrame(0, index=np.arange(1), columns=feature_names) # new df - same cols as training data & filling missing with 0
    for col in encoded_input_df.columns:
        if col in final_input_df.columns:
            final_input_df[col] = encoded_input_df[col]
    
    return final_input_df

def predict_target_rate(model, processed_input, X_test, y_test=None):
    prediction = model.predict(processed_input)
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    accuracy = r2 * 100
    return prediction[0], accuracy

if __name__ == "__main__":
    model_filename = "trained_model.pkl"
    encoder_filename = "encoder.pkl"
    test_data_filename = "test_data.pkl"

    model, encoder, X_test, y_test = load_model_and_encoder(model_filename, encoder_filename, test_data_filename) # load

    feature_names = encoder.get_feature_names_out(['equipment_type', 'origin_city', 'origin_state', 'destination_city', 'destination_state', 'day', 'month', 'year']) # features

    # input
    equipment_type = input("Enter equipment type (1 or 2): ")
    while equipment_type not in ['1', '2']:
        print("Invalid input. Please enter 1 or 2.")
        equipment_type = input("Enter equipment type (1 or 2): ")

    while True:
        origin = input("Enter origin (city, state): ")
        try:
            origin_city, origin_state = origin.split(',')
            origin_city = origin_city.strip()
            origin_state = origin_state.strip()

            if (origin_city, origin_state) not in valid_origin:
                print("Invalid input. Please enter a valid city and state combination.")
            else:
                break
        except ValueError:
            print("Invalid input format. Please enter in the format 'city, state'.")

    while True:
        destination = input("Enter destination (city, state): ")
        try:
            destination_city, destination_state = destination.split(',')
            destination_city = destination_city.strip()
            destination_state = destination_state.strip()

            if (destination_city, destination_state) not in valid_destination:
                print("Invalid input. Please enter a valid city and state combination.")
            else:
                break
        except ValueError:
            print("Invalid input format. Please enter in the format 'city, state'.")

    date_input = input("Enter date (dd/mm/yyyy): ")
    day, month, year = date_input.split('/')
    day = int(day)
    month = int(month)
    year = int(year)
    while not (1 <= day <= 31) or not (1 <= month <= 12):
        print("Invalid date input. Please enter date in dd/mm/yyyy format.")
        date_input = input("Enter date (dd/mm/yyyy): ")
        day, month, year = date_input.split('/')
        day = int(day)
        month = int(month)
        year = int(year)

    processed_input = preprocess_input(equipment_type, origin_city, origin_state, destination_city, destination_state, day, month, year, encoder, feature_names) # process

    predicted_rate, accuracy = predict_target_rate(model, processed_input, X_test, y_test) # prediction

    print(f"Predicted target rate: {predicted_rate:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")