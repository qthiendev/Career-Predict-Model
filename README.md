
# Career-Predict-Model

## About

### Author

> by [Trinh Quy Thien](https://www.linkedin.com/in/qthiendev/) a.k.a [qthiendev](https://github.com/qthiendev)

Special thanks to [Mrs. Nguyen Thi Thu Suong](https://github.com/thusgthusg) and [Mr. Truong Dinh Huy](https://is.duytan.edu.vn/international-school/71-truong-dinh-huy) for their guidance and support in creating this model.

This model is part of the **Career Recommendation Module** of the [NavCareer | Career Support System](https://github.com/qthiendev/NavCareer_C1SE.15_12-2024), a Capstone 1 Project for CMU-SE 450 AIS, Duy Tan University.

The model can work independently or be modified to fit other systems.

---

## Project

The **Career-Predict-Model** is designed to predict suitable careers based on user responses. It features:

- **Career Prediction**: Uses a trained Random Forest Classifier to suggest careers.
- **Dynamic Updates**: Supports adding new career entries and user responses to the dataset.
- **Custom Training**: Includes functionality to retrain the model with updated datasets.
- **Flexibility**: Can be integrated into other career guidance or recommendation systems.

---

## Usage Guide

### 1. Installation

1. **Python Setup**:
   - The model is developed using Python 3.13.1. [Download the latest version here](https://www.python.org/downloads/).
   - During installation, make sure to check **Add to PATH**.

2. **Clone the Project**:
   - Open the command prompt (`cmd`) and navigate to the desired directory.
   - Clone this project using the following command:

     ```bash
     git clone https://github.com/qthiendev/Career-Predict-Model.git ./Career-Predict-Model
     ```

3. **Install Required Libraries**:
   - Run the following command to install the required Python libraries:

     ```bash
     pip install pandas scikit-learn imbalanced-learn joblib logging numpy
     ```

---

### 2. Directory Structure

    Career-Predict-Model/
    │
    ├── data/
    │   ├── v1.0/                     # Initial data version
    │   ├── v1.1/
    │   └── v1.2/                     # Current version (used by default)
    │       ├── archetype.csv         # Archetype definitions
    │       ├── careers.csv           # List of career codes and names
    │       ├── responses.csv         # User responses dataset
    │       ├── dominance.csv         # Dominance attributes
    │       ├── question.txt          # Questions for input
    │       └── Visualize.xlsx        # Visualization file for analysis
    │
    ├── models/
    │   ├── v1.0/                     # Old models
    │   ├── v1.1/                     # Intermediate model
    │   └── v1.2/                     # Latest trained model
    │       ├── career_predictor_model.pkl
    │       └── training.log
    │
    ├── old_version/
    │   └── train_model_v1.0.py       # Initial training script
    │
    ├── LICENSE                       # License file
    ├── README.md                     # Documentation
    ├── train_model_v1.1.py           # Current training script
    ├── predict_career.py             # Prediction script
    ├── nodejs_predict.py             # Node.js integration for prediction
    └── nodejs_predict_save.py        # Node.js integration for saving responses

---

### 3. How to Use

#### **Prediction**

1. To predict a career, run the `predict_career.py` script:

   ```bash
   python predict_career.py
   ```

2. Enter 30 responses to the questions displayed in `data/v1.2/question.txt` (e.g., `{1, 2, 5, 4, ...}`).

3. The system will output the predicted career name and code.

#### **Adding New Careers**

1. If the predicted career is incorrect, you can provide the correct career name.

2. The script will dynamically update `careers.csv` and `responses.csv` with the new entry.

#### **Training**

1. To retrain the model with updated datasets, use the `train_model_v1.1.py` script:

   ```bash
   python train_model_v1.1.py
   ```

2. The new trained model will be saved in the `models/v1.2/` directory.

---

### 4. Logging

The model logs its training and prediction processes in `training.log` located in the `models/v1.2/` directory. Logs include:

- Dataset loading times.
- Hyperparameter tuning and model evaluation.
- Errors and warnings (if any).

---

### 5. Customization

- **Dataset Versions**: The model uses `v1.2` data by default. To switch to another version, update the `model_version` variable in the scripts.
- **Questions**: Modify `question.txt` to customize the input questionnaire.
- **Careers**: Update `careers.csv` to include new career codes or names.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
