from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import re
import numpy as np
from io import StringIO


model = joblib.load('elastic_net_model.pkl')
median_values = joblib.load('median_values.pkl')
scaler = joblib.load('scaler.pkl')
median_values = dict(median_values)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def preprocess_data(data: List[Item], median_values: dict) -> pd.DataFrame:
    df = pd.DataFrame([item.dict() for item in data])
    df = df.drop_duplicates(subset=df.columns, keep='first')
    df.reset_index(drop=True, inplace=True)

    unit_replacements = {
        'mileage': r' kmpl| km/kg',
        'engine': r' CC',
        'max_power': r' bhp'
    }
    for column, pattern in unit_replacements.items():
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(pattern, '', regex=True),
            errors='coerce'
        )

    torque_values = []
    rpm_values = []
    for torque in df['torque']:
        torque_str = str(torque)

        extracted_torques = []
        extracted_rpms = []

        torque_matches = re.findall(r'([\d.]+)\s*(?:-|and)?\s*([\d.]*)\s*(Nm|kgm)', torque_str, re.IGNORECASE)
        for match in torque_matches:
            torque_1 = float(match[0])
            torque_2 = float(match[1]) if match[1] else torque_1
            max_torque = max(torque_1, torque_2)
            unit = match[2].lower()
            if unit == 'kgm':
                max_torque *= 9.8
            extracted_torques.append(max_torque)

        max_torque = max(extracted_torques) if extracted_torques else None

        rpm_matches = re.findall(r'@?\s*([\d,.-]+)\s*rpm', torque_str, re.IGNORECASE)
        for rpm in rpm_matches:
            rpm_clean = rpm.replace(',', '').replace('.', '').strip()
            if '-' in rpm_clean:
                rpm_range = [int(r) for r in rpm_clean.split('-') if r.strip().isdigit()]
                extracted_rpms.extend(rpm_range)
            elif rpm_clean.isdigit():
                extracted_rpms.append(int(rpm_clean))

        max_rpm = max(extracted_rpms) if extracted_rpms else None

        torque_values.append(max_torque)
        rpm_values.append(max_rpm)

    df['torque'] = torque_values
    df['max_torque_rpm'] = rpm_values

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(median_values)

    columns_to_convert = ['engine', 'seats']
    df[columns_to_convert] = df[columns_to_convert].astype(np.int64)

    df = df[numeric_columns]

    df_scaled = scaler.transform(df)

    return df_scaled


@app.post("/predict_item")
def predict_item(item: Item):
    processed_data = preprocess_data([item], median_values)
    prediction = model.predict(processed_data)

    return {"predicted_price": float(prediction[0])}


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    items = df.to_dict(orient='records')
    processed_data = preprocess_data([Item(**item) for item in items], median_values)

    predictions = model.predict(processed_data)
    df['predicted_price'] = [float(p) for p in predictions]
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return {"file": output.getvalue()}
