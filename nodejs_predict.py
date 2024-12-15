import sys
import json
import pandas as pd
import os
import joblib
from concurrent.futures import ThreadPoolExecutor

MODEL_VERSION = 'v1.2'

def load_resources(base_dir, model_version):
    """Load resources concurrently."""
    career_codes_path = os.path.join(base_dir, 'data', model_version, 'careers.csv')
    model_path = os.path.join(base_dir, 'models', model_version, 'career_predictor_model.pkl')

    def load_model():
        return joblib.load(model_path, mmap_mode='r')

    def load_career_codes():
        return pd.read_csv(career_codes_path, usecols=['Code', 'Career Name', 'Career Name (Vietnamese)'])

    with ThreadPoolExecutor() as executor:
        future_model = executor.submit(load_model)
        future_career_codes = executor.submit(load_career_codes)

        model = future_model.result()
        career_codes = future_career_codes.result()

    return model, career_codes

def main():
    try:
        base_dir = os.path.dirname(__file__)
        model, career_codes = load_resources(base_dir, MODEL_VERSION)

        input_data = json.loads(sys.argv[1])

        if len(input_data) != 30:
            raise ValueError("Please ensure exactly 30 responses are entered.")

        predicted_code = model.predict([input_data])[0]

        career_row = career_codes.loc[career_codes['Code'] == predicted_code].iloc[0]
        career_name = career_row['Career Name']
        career_name_vie = career_row.get('Career Name (Vietnamese)', "")

        output = f"{predicted_code}, {career_name}, {career_name_vie}"
        sys.stdout.buffer.write(output.encode('utf-8'))

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
