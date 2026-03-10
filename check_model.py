import pickle
import numpy as np

# Load model
with open('trained_model.pkl', 'rb') as f:
    data = pickle.load(f)

print("Model contents:")
print(f"Keys: {data.keys()}")
print(f"Number of encodings: {len(data['encodings'])}")
print(f"Names: {set(data['names'])}")
print(f"Feature dimension: {len(data['encodings'][0])}")
print(f"First encoding sample: {data['encodings'][0][:10]}")
print(f"Data type: {type(data['encodings'][0])}")