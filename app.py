from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn, pickle
from main2 import load_model_and_encoder, preprocess_input, predict_target_rate

with open('valid_origin.pkl', 'rb') as f:
    valid_origin = pickle.load(f)

with open('valid_destination.pkl', 'rb') as f:
    valid_destination = pickle.load(f)

model_filename = "trained_model.pkl"
encoder_filename = "encoder.pkl"
test_data_filename = "test_data.pkl"
model, encoder, X_test, y_test = load_model_and_encoder(model_filename, encoder_filename, test_data_filename)

app = FastAPI() # FastAPI app

class InputData(BaseModel): #input
    equipment_type: str
    origin: str
    destination: str
    date: str

    @validator('equipment_type') # equipment type validation
    def validate_equipment_type(cls, v):
        if v not in ['1', '2']:
            raise ValueError("Equipment type - '1' or '2'")
        return v

    @validator('date') # date validation
    def validate_date_format(cls, v):
        try:
            day, month, year = map(int, v.split('/'))
            if not (1 <= day <= 31 and 1 <= month <= 12):
                raise ValueError("Invalid date format")
        except ValueError:
            raise ValueError("Invalid - 'dd/mm/yyyy' format required.")
        return v

    @validator('destination', pre=True, always=True) # origin neq destination
    def validate_different_origin_and_destination(cls, v, values, **kwargs):
        if 'origin' in values and v == values['origin']:
            raise ValueError("Origin and destination cannot be the same.")
        return v

def validate_origin_and_destination(input_data: InputData): # origin and destination validation
    origin_city, origin_state = input_data.origin.split(',')
    destination_city, destination_state = input_data.destination.split(',')
    if (origin_city.strip(), origin_state.strip()) not in valid_origin:
        raise HTTPException(status_code=400, detail="Invalid origin")
    if (destination_city.strip(), destination_state.strip()) not in valid_destination:
        raise HTTPException(status_code=400, detail="Invalid destination")
    return input_data

@app.post("/predict_rate") # POST endpoint for prediction
async def predict_rate(input_data: InputData = Depends(validate_origin_and_destination)):
    try:
        origin_city, origin_state = input_data.origin.split(',')
        destination_city, destination_state = input_data.destination.split(',')
        day, month, year = map(int, input_data.date.split('/'))

        processed_input = preprocess_input(input_data.equipment_type, origin_city.strip(), origin_state.strip(),
                                           destination_city.strip(), destination_state.strip(),
                                           day, month, year, encoder, encoder.get_feature_names_out(['equipment_type', 'origin_city', 'origin_state', 'destination_city', 'destination_state', 'day', 'month', 'year']))

        predicted_rate, accuracy = predict_target_rate(model, processed_input, X_test, y_test)

        predicted_rate = round(predicted_rate)
        accuracy = round(accuracy)

        return JSONResponse(content={
            "predicted_rate": predicted_rate,
            "accuracy": f"{accuracy}%"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    print("server shuru...")
    uvicorn.run(app, host="127.0.0.1", port=8000)