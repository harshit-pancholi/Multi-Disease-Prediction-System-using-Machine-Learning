import pickle
import numpy as np

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict(model, input_data):

    input_data = np.array(input_data).reshape(1, -1)

    if input_data.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Expected {model.n_features_in_} features, got {input_data.shape[1]}"
        )

    try:
        prediction = model.predict(input_data)
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

    return prediction[0], probability