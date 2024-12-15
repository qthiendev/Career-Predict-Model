import pandas as pd
import numpy as np
import random
import os

model_version = 'v1.2'
data_dir = os.path.join(os.path.dirname(__file__), model_version)

careers_path = os.path.join(data_dir, 'careers.csv')
archetype_path = os.path.join(data_dir, 'archtype.csv')
responses_path = os.path.join(data_dir, 'responses.csv')

for path in [careers_path, archetype_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file '{path}' not found.")

career_archetypes = pd.read_csv(careers_path, index_col=0).iloc[:, 0].to_dict()
career_codes = list(career_archetypes.keys())
num_careers = len(career_codes)

archetype_df = pd.read_csv(archetype_path, index_col=0)
archetype_patterns = {
    row.name: [row[f"Q{i+1}_A{j+1}"] / 100 for i in range(30) for j in range(5)]
    for _, row in archetype_df.iterrows()
}

num_samples = 18000 - 9000
question_count = 30
generated_data = []

samples_per_career = num_samples // num_careers
remaining_samples = num_samples % num_careers

for career_code in career_codes:
    print('Answer for', career_code, 'creating.')
    for _ in range(samples_per_career):
        pattern = archetype_patterns.get(career_code, [0.2] * 150)
        question_patterns = [pattern[i*5:(i+1)*5] for i in range(question_count)]
        normalized_patterns = [np.array(p) / np.sum(p) for p in question_patterns]
        answers = [np.random.choice([1, 2, 3, 4, 5], p=normalized_patterns[i]) for i in range(question_count)]
        generated_data.append(answers + [career_code])
    print('Answer for', career_code, 'created.')

for _ in range(remaining_samples):
    career_code = random.choice(career_codes)
    pattern = archetype_patterns.get(career_code, [0.2] * 150)
    question_patterns = [pattern[i*5:(i+1)*5] for i in range(question_count)]
    normalized_patterns = [np.array(p) / np.sum(p) for p in question_patterns]
    answers = [np.random.choice([1, 2, 3, 4, 5], p=normalized_patterns[i]) for i in range(question_count)]
    generated_data.append(answers + [career_code])

responses_df = pd.DataFrame(generated_data, columns=[f"Q{i+1}" for i in range(question_count)] + ["Career_Code"])

if os.path.exists(responses_path):
    responses_df.to_csv(responses_path, mode='a', header=False, index=False)
else:
    responses_df.to_csv(responses_path, mode='w', header=True, index=False)

print(f"{num_samples} Data appended to '{responses_path}' successfully.")
