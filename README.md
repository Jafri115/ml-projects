# Student Exam Performance Prediction

This project predicts students' mathematics exam performance based on various factors, such as gender, ethnicity, parental education level, and preparation courses.

## Project Structure

```
ML-PROJECTS/
│
├── artifacts/                # Saved models and preprocessor files
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── test.csv
│   ├── train.csv
│
├── logs/                     # Log files
│
├── notebook/                 # Jupyter notebooks for analysis
│   └── 1. EDA STUDENT PERFORMANCE.ipynb
│
├── src/                      # Source code for the project
│   ├── data_ingestion.py     # Handles loading and splitting data
│   ├── data_transformation.py # Preprocessing data
│   ├── model_trainer.py      # Model training and evaluation
│
├── templates/                # HTML templates for web application
│   ├── home.html
│   ├── index.html
│
├── .gitignore                # Files and directories to ignore in Git
├── application.py            # Flask application for predictions
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Project setup file
```

## Key Components

### 1. Data Ingestion
The `data_ingestion.py` script reads and splits the data into training and testing sets.

### 2. Data Transformation
`data_transformation.py` handles:
- Encoding categorical variables
- Scaling numerical features
- Saving preprocessing objects

### 3. Model Training
`model_trainer.py` trains multiple models (e.g., Random Forest, Gradient Boosting) and evaluates their performance using R².

### 4. Web Interface
`application.py` uses Flask to create a web interface for predicting math scores based on user inputs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-performance-predictor.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python application.py
   ```

## How to Use

1. Open the web application in your browser at `http://127.0.0.1:5000/`.
2. Fill out the form with the required details.
3. Submit the form to predict the math score.

 ![Screenshot of Web Application](img\screenshot_webpage.png)

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework
- **Scikit-learn**: Machine learning library
- **Pandas** and **NumPy**: Data manipulation
- **CatBoost** and **XGBoost**: Advanced regression models
- **HTML/CSS**: Frontend development

## AWS Deployment

This project is deployed on **AWS Elastic Beanstalk** with continuous integration using **AWS CodePipeline**.

### Steps for Deployment:

1. **Set Up Elastic Beanstalk Environment**:
   - Create an Elastic Beanstalk application.
   - Set up a Python-based environment.
  

2. **Prepare CodePipeline**:
   - Create an S3 bucket for storing deployment artifacts.
   - Set up a CodePipeline with the following stages:
     - **Source**: Connect the pipeline to your GitHub repository.
     - **Build**: Use AWS CodeBuild to prepare the application for deployment.
     - **Deploy**: Deploy the application to Elastic Beanstalk.

3. **Update Application Configurations**:
   - Add necessary IAM roles and permissions for Elastic Beanstalk and CodePipeline.
   - Ensure that required environment variables (if any) are configured in Elastic Beanstalk.

4. **Trigger Deployment**:
   - Push your code to the GitHub repository to trigger the pipeline.
   - Monitor the pipeline execution and Elastic Beanstalk environment.

## Project Pipeline

1. **Data Loading and Cleaning**
2. **Exploratory Data Analysis (EDA)** using Jupyter Notebook
3. **Feature Engineering and Preprocessing**
4. **Model Training and Evaluation**
5. **Web Deployment** (using AWS)