# Loan Prediction System

A web application built with Streamlit that helps predict loan approval based on applicant information using machine learning algorithms.

![Loan Prediction System](https://img.shields.io/badge/Loan%20Prediction-System-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)

## Features

- **Data Analysis**: Explore and visualize the loan dataset
- **Model Training**: Train and evaluate multiple machine learning models
- **Prediction**: Make individual and batch predictions for loan approval
- **Interactive UI**: User-friendly interface with visualizations and explanations

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Models](#models)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/loan-prediction-system.git
   cd loan-prediction-system
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

The application has three main sections:

### 1. Data Analysis

- View and explore the loan dataset
- Analyze data distributions and correlations
- Understand key factors affecting loan approval
- Visualize relationships between features and loan status

### 2. Model Training

- Preprocess data by handling missing values and encoding categorical features
- Train multiple machine learning models (Random Forest, Logistic Regression, etc.)
- Compare model performances and select the best one
- Analyze feature importance

### 3. Prediction

- Individual prediction: Enter applicant details to get loan approval prediction
- Batch prediction: Upload a CSV file with multiple applicants' data
- View prediction results with approval probabilities
- Download prediction results

## Project Structure

```
loan-prediction-system/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Required Python packages
├── README.md           # Project documentation
└── data/               # Dataset folder (created when app is first run)
```

## Data Description

The loan prediction dataset contains the following features:

| Feature            | Description                                      |
|--------------------|--------------------------------------------------|
| Gender             | Gender of the applicant (Male/Female)            |
| Married            | Marital status (Yes/No)                          |
| Dependents         | Number of dependents (0, 1, 2, 3+)               |
| Education          | Education level (Graduate/Not Graduate)          |
| Self_Employed      | Self-employment status (Yes/No)                  |
| ApplicantIncome    | Monthly income of the applicant                  |
| CoapplicantIncome  | Monthly income of the co-applicant               |
| LoanAmount         | Loan amount (in thousands)                       |
| Loan_Amount_Term   | Loan term in months                              |
| Credit_History     | Credit history (1: Good, 0: Bad)                 |
| Property_Area      | Area of property (Urban/Semiurban/Rural)         |
| Loan_Status        | Loan approval status (Y: Approved, N: Rejected)  |

## Models

The application trains and evaluates the following machine learning models:

1. **Logistic Regression**: A statistical model for binary classification
2. **Random Forest**: An ensemble of decision trees
3. **Gradient Boosting**: A boosting algorithm that builds trees sequentially
4. **Support Vector Machine**: A classifier that finds the optimal hyperplane

## Screenshots

[Include screenshots of your application here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
