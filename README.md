# Machine Learning Project: Building and Deploying with Modular Structure

This project demonstrates the full lifecycle of building, structuring, and deploying a machine learning solution. It follows best practices in modular design, cloud deployment, and automation to ensure scalability and maintainability.

---

## Key Takeaways & Learnings

- **Structured Approach:** The project adopts a modular structure that separates data ingestion, transformation, model training, and deployment components. This organization allows for clear, maintainable, and reusable code across various ML tasks.

---

## Project Workflow

### Step 1: Manage Dependencies

The project uses a `requirements.txt` file to manage dependencies:

```bash
numpy
pandas
scikit-learn
logging
```

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

---

### Step 2: Organize Project Structure

The project follows a structured directory layout:

```
ml_project/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── exception.py
│   │   ├── utils.py
├── setup.py
├── requirements.txt
└── README.md
```

**`setup.py`** file for packaging the project:

```python
from setuptools import setup, find_packages

setup(
    name="ml_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[]  # Add dependencies here
)
```

---

### Step 3: Implementing ML Components

- **Data Ingestion:** Loads data using `pandas`.

```python
import pandas as pd

class DataIngestion:
    def load_data(self, file_path):
        return pd.read_csv(file_path)
```

- **Data Transformation:** Preprocesses data with `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def transform(self, data):
        le = LabelEncoder()
        data['category'] = le.fit_transform(data['category'])
        return data
```

- **Model Training:** Trains and evaluates the model using `RandomForestClassifier` and `accuracy_score`.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def train(self, X_train, y_train):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    
    def evaluate(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)
```

---

### Step 4: Implementing Pipelines

- **Training Pipeline:** Manages the end-to-end training process.

```python
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer

class TrainPipeline:
    def run(self):
        ingestion = DataIngestion()
        data = ingestion.load_data('data.csv')
        transformer = DataTransformation()
        transformed_data = transformer.transform(data)
        trainer = ModelTrainer()
        model = trainer.train(transformed_data.drop('target', axis=1),
                              transformed_data['target'])
        accuracy = trainer.evaluate(model, X_test, y_test)
        print(f"Model Accuracy: {accuracy}")
```

- **Prediction Pipeline:** Handles predictions with a pre-trained model.

```python
from joblib import load

class PredictPipeline:
    def predict(self, data):
        model = load('model.joblib')
        return model.predict(data)
```

---

### Step 5: Automating CI/CD with GitHub Actions and Docker

The next phase of the project focuses on automating the build and deployment process using GitHub Actions to push the Docker image to AWS EC2 once the build is complete.

**Dockerfile:**

This Dockerfile creates a lightweight image with Python dependencies and project code:

```Dockerfile
FROM python:3.8-slim-buster

WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt

CMD ["python3", "app.py"]
```

---

### Step 6: Logging and Exception Handling

- **Logging:** Creates a `logger.py` file for managing logging.

```python
import logging
import os

log_file = os.path.join(os.getcwd(), "logs", "project.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def log_message(message):
    logging.info(message)
```

- **Exception Handling:** Defines custom exceptions in `exception.py`.

```python
class CustomException(Exception):
    def __init__(self, message, error):
        super().__init__(message)
        self.error = error
```

---

### Step 7: Utilities

- **Utility Functions:** Provides reusable functions in `utils.py` for saving and loading models.

```python
from joblib import dump, load

def save_model(model, file_path):
    dump(model, file_path)

def load_model(file_path):
    return load(file_path)
```

---

### Step 8: Testing

- **Component Testing:** Ensure individual components like data ingestion and model training work correctly.
- **Pipeline Testing:** Run `train_pipeline.py` to test the entire training pipeline.
- **Logging and Exceptions:** Verify that all logs and exceptions are captured correctly.

---

### Step 9: CI/CD Pipeline Configuration

The GitHub Actions workflow file automates the build and deployment process. This includes building the Docker image, pushing it to Amazon ECR, and deploying it on an AWS EC2 machine.

**GitHub Actions Workflow:**

 

---
 
