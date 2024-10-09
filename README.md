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
- **Data Transformation:** Preprocesses data with `LabelEncoder`.
- **Model Training:** Trains and evaluates the model using `RandomForestClassifier` and `accuracy_score`.

---

### Step 4: Automating CI/CD with GitHub Actions and Docker

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

### Step 5: CI/CD Pipeline Configuration

The GitHub Actions workflow file automates the build and deployment process. This includes building the Docker image, pushing it to Amazon ECR, and deploying it on an AWS EC2 machine.


---

### Step 6: Project Deployment

1. **Build Docker Image:** The GitHub Action workflow checks out the latest code, installs utilities, builds a Docker image, and pushes it to Amazon Elastic Container Registry (ECR).
2. **Deploy to AWS EC2:** After building the image, the workflow pulls the latest image to the EC2 instance and runs the containerized ML application, making it available to users.
3. **Cleanup:** After deployment, the workflow removes any unused Docker containers and images to free up space.

---

## Technical Skills Highlighted

- Python programming
- Machine learning libraries (`scikit-learn`)
- Data manipulation (`pandas`, `numpy`)
- Web development (Flask)
- CI/CD with GitHub Actions
- Docker image creation and management
- Cloud deployment (AWS EC2, Amazon ECR)

---

## Project Impact

This project provided a comprehensive, hands-on experience in structuring and deploying a machine learning project. By implementing GitHub Actions for CI/CD, Docker for containerization, and AWS EC2 for deployment, the project achieved automation, scalability, and efficiency. This approach equips me with the skills to tackle real-world machine learning tasks and deployment pipelines.

--- 
