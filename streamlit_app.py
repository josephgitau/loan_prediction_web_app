# importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
import io
import base64

# page config
st.set_page_config(page_title='Loan Prediction App', layout='wide')

# Load data function
@st.cache_data
def load_data():
    # Try to load the dataset with loan status values
    # Original URL that seems to have missing target values
    st.success("Loading data...")
    url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
    # Set index_col=0 to use the first unnamed column as index
    df = pd.read_csv(url, index_col=0)
            
    return df

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None
if 'fitted_encoder' not in st.session_state:
    st.session_state.fitted_encoder = None
if 'cat_columns' not in st.session_state:
    st.session_state.cat_columns = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'models_comparison' not in st.session_state:
    st.session_state.models_comparison = None

# Navigation
def navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Model Training", "Prediction"])
    return page

# Preprocess data function
def preprocess_data(df):
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    processed_df['Gender'].fillna(processed_df['Gender'].mode()[0], inplace=True)
    processed_df['Married'].fillna(processed_df['Married'].mode()[0], inplace=True)
    processed_df['Dependents'].fillna(processed_df['Dependents'].mode()[0], inplace=True)
    processed_df['Self_Employed'].fillna(processed_df['Self_Employed'].mode()[0], inplace=True)
    processed_df['LoanAmount'].fillna(processed_df['LoanAmount'].median(), inplace=True)
    processed_df['Loan_Amount_Term'].fillna(processed_df['Loan_Amount_Term'].mode()[0], inplace=True)
    processed_df['Credit_History'].fillna(processed_df['Credit_History'].mode()[0], inplace=True)
    
    # Ensure Loan_Status is properly preserved
    # Convert 'Y'/'N' values to 1/0 explicitly to avoid missing values later
    if 'Loan_Status' in processed_df.columns:
        # Handle different formats of Loan_Status
        if processed_df['Loan_Status'].dtype == 'object':
            # Try mapping common values
            status_map = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'Approved': 1, 'Rejected': 0}
            processed_df['Loan_Status'] = processed_df['Loan_Status'].map(status_map)
        else:
            # If already numeric, ensure it's 0 or 1
            processed_df['Loan_Status'] = processed_df['Loan_Status'].astype(int)
    
    # Encode categorical features
    encoder = OrdinalEncoder()
    categorical_cols = ['Gender', 'Married', 'Education', 'Dependents', 'Self_Employed', 'Property_Area']
    processed_df[categorical_cols] = encoder.fit_transform(processed_df[categorical_cols])
    
    # Save the encoder and categorical columns for later use
    st.session_state.fitted_encoder = encoder
    st.session_state.cat_columns = categorical_cols
    
    return processed_df

# Function to create download link for dataframe
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {text}</a>'
    return href

# Function to preprocess user input for prediction
def preprocess_input(data):
    # Create a DataFrame from user input
    input_df = pd.DataFrame(data, index=[0])
    
    # Convert string values to appropriate numeric types
    if 'LoanAmount' in input_df:
        input_df['LoanAmount'] = input_df['LoanAmount'].astype(float)
    if 'Loan_Amount_Term' in input_df:
        input_df['Loan_Amount_Term'] = input_df['Loan_Amount_Term'].astype(float)
    if 'Credit_History' in input_df:
        input_df['Credit_History'] = input_df['Credit_History'].astype(int)
    
    # Use the fitted encoder to transform categorical features
    categorical_data = input_df[st.session_state.cat_columns]
    input_df[st.session_state.cat_columns] = st.session_state.fitted_encoder.transform(categorical_data)
    
    return input_df

# Function to train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    results = {}
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Store results
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores)
        }
        
        # Track best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
    
    return results, best_model

# Data Analysis Page
def data_analysis_page():
    st.title("Loan Prediction System - Data Analysis")
    
    st.write("""
    ## Welcome to the Loan Prediction System
    
    This application helps predict whether a loan application will be approved based on various applicant features.
    
    ### About the Dataset
    The dataset contains information about loan applicants including their personal details, income, loan amount, and whether their loan was approved.
    """)
    
    df = st.session_state.df
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())
    
    # Dataset info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of Records:** {df.shape[0]}")
        st.write(f"**Number of Features:** {df.shape[1] - 1}")  # Excluding target variable
    with col2:
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
        # Fix for the error - check what values are in Loan_Status
        loan_status_counts = df['Loan_Status'].value_counts(normalize=True)
        if 'Y' in loan_status_counts.index:
            approval_rate = loan_status_counts['Y'] * 100
        else:
            # If 'Y' is not found, try to find the most common value or use the first value
            approval_rate = loan_status_counts.iloc[0] * 100
            
        st.write(f"**Loan Approval Rate:** {approval_rate:.2f}%")
        
        # Fix for TypeError: Use value_counts instead of unique to display loan status values
        st.write("**Loan Status Distribution:**")
        st.dataframe(df['Loan_Status'].value_counts().reset_index().rename(
            columns={'index': 'Status', 'Loan_Status': 'Count'}))
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_data = pd.DataFrame({
        'Feature': df.columns,
        'Missing Values': df.isnull().sum().values,
        'Percentage': (df.isnull().sum().values / len(df) * 100)
    })
    st.dataframe(missing_data.sort_values('Missing Values', ascending=False))
    
    # Data distribution
    st.subheader("Data Distribution")
    
    # Categorical features analysis
    st.write("### Categorical Features")
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Distribution", "Relationship with Target", "Correlation"])
    
    with tab1:
        # Display distribution of categorical variables
        for i in range(0, len(categorical_cols)-1, 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(categorical_cols)-1:  # Exclude Loan_Status from this loop
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        df[categorical_cols[i+j]].value_counts().plot(kind='bar', ax=ax)
                        plt.title(f'Distribution of {categorical_cols[i+j]}')
                        plt.tight_layout()
                        st.pyplot(fig)
    
    with tab2:
        # Display relationship with target variable
        for i in range(0, len(categorical_cols)-1, 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(categorical_cols)-1:
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        pd.crosstab(df[categorical_cols[i+j]], df['Loan_Status']).plot(kind='bar', ax=ax)
                        plt.title(f'{categorical_cols[i+j]} vs Loan Status')
                        plt.tight_layout()
                        st.pyplot(fig)
    
    with tab3:
        # Correlation heatmap
        st.write("### Feature Correlation")
        # Convert categorical to numeric for correlation
        df_encoded = df.copy()
        
        # Check if Loan_ID column exists and drop it before correlation
        if 'Loan_ID' in df_encoded.columns:
            df_encoded = df_encoded.drop('Loan_ID', axis=1)
        
        # Convert remaining categorical columns to numeric
        for col in categorical_cols:
            if col in df_encoded.columns:  # Only process columns that exist
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        # Ensure all columns are numeric before correlation
        numeric_df = df_encoded.select_dtypes(include=[np.number])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Numerical features analysis
    st.write("### Numerical Features")
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Distribution", "Relationship with Target"])
    
    with tab1:
        # Display distribution of numerical variables
        for i in range(0, len(numerical_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(numerical_cols):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(df[numerical_cols[i+j]].dropna(), kde=True, ax=ax)
                        plt.title(f'Distribution of {numerical_cols[i+j]}')
                        plt.tight_layout()
                        st.pyplot(fig)
    
    with tab2:
        # Display relationship with target variable
        for i in range(0, len(numerical_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(numerical_cols):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.boxplot(x='Loan_Status', y=numerical_cols[i+j], data=df, ax=ax)
                        plt.title(f'{numerical_cols[i+j]} vs Loan Status')
                        plt.tight_layout()
                        st.pyplot(fig)
    
    # Key insights
    st.subheader("Key Insights")
    st.write("""
    1. **Credit History** appears to be the most important factor in loan approval.
    2. Applicants with higher income tend to have higher approval rates.
    3. Married applicants have slightly higher approval rates than unmarried ones.
    4. Property area seems to influence loan approval, with semi-urban areas having higher approval rates.
    5. Education level shows some correlation with loan approval, with graduates having better chances.
    """)
    
    # Download option
    st.subheader("Download Data")
    st.markdown(get_table_download_link(df, "loan_data.csv", "Loan Data as CSV"), unsafe_allow_html=True)

# Model Training Page
def model_training_page():
    st.title("Model Training and Evaluation")
    
    df = st.session_state.df
    
    # Data preprocessing
    st.subheader("Data Preprocessing")
    
    with st.expander("Preprocessing Steps", expanded=True):
        st.write("""
        1. **Handling Missing Values**:
           - Categorical features: filled with mode
           - Numerical features: filled with median
           - Target variable (Loan_Status): rows with missing values are removed
        
        2. **Feature Encoding**:
           - Categorical features encoded using Ordinal Encoding
        
        3. **Feature Selection**:
           - All relevant features used for prediction
        """)
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                # Check if Loan_Status column exists and has values before preprocessing
                if 'Loan_Status' not in df.columns:
                    st.error("Error: No 'Loan_Status' column found in the dataset. Cannot train a model.")
                elif df['Loan_Status'].isnull().all():
                    st.error("Error: All values in the target column (Loan_Status) are missing. Cannot train a model.")
                else:
                    # First, check how many non-null values we have in Loan_Status
                    non_null_count = df['Loan_Status'].count()
                    total_count = len(df)
                    st.info(f"Target column has {non_null_count} non-null values out of {total_count} total rows.")
                    
                    # Proceed with preprocessing
                    st.session_state.preprocessed_df = preprocess_data(df)
                    
                    # Verify we still have data after preprocessing
                    if len(st.session_state.preprocessed_df) == 0:
                        st.error("Error: No data remains after preprocessing. Check your data for missing values.")
                        st.session_state.preprocessed_df = None
                    else:
                        st.success(f"Data preprocessing completed! Retained {len(st.session_state.preprocessed_df)} rows.")
    
    # Display preprocessed data if available
    if st.session_state.preprocessed_df is not None and len(st.session_state.preprocessed_df) > 0:
        st.subheader("Preprocessed Data")
        st.dataframe(st.session_state.preprocessed_df.head())
        
         # Add option to show model training data
        with st.expander("Show Model Training Data"):
            # Prepare data for modeling
            X = st.session_state.preprocessed_df.copy()
            
            # Handle the target variable properly
            if 'Loan_Status' in X.columns:
                y = X['Loan_Status']
                X = X.drop('Loan_Status', axis=1)

                # Display distribution of target variable
                st.subheader("Target Variable Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                y_counts = y.value_counts()
                sns.barplot(x=y_counts.index, y=y_counts.values, ax=ax)
                plt.title('Distribution of Loan Status')
                plt.xticks([0, 1], ['Rejected (0)', 'Approved (1)'])
                plt.tight_layout()
                st.pyplot(fig)

                # Check for any null values in the target variable
                null_count = y.isnull().sum()
                if null_count > 0:
                    st.warning(f"Warning: {null_count} missing values detected in the target variable.")
            else:
                st.error("Error: No 'Loan_Status' column found in the preprocessed data.")
                y = None

            # Remove Loan_ID if it exists
            if 'Loan_ID' in X.columns:
                X = X.drop('Loan_ID', axis=1)
            
            # Show features for training
            st.subheader("Features for Model Training")
            st.dataframe(X.head())
            st.write(f"Shape of feature data: {X.shape}")

            # Check for any remaining null values in features
            null_cols = X.columns[X.isnull().any()].tolist()
            if null_cols:
                st.warning(f"Warning: Null values detected in these columns: {', '.join(null_cols)}")

            # Display training/test split preview
            if y is not None and not y.isnull().all():
                test_size = st.slider("Test Set Size (%)", 10, 40, 20, key="preview_split_slider") / 100
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Training set: {X_train.shape[0]} samples")
                        st.write(f"Approved loans: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train):.1%})")
                    with col2:
                        st.write(f"Test set: {X_test.shape[0]} samples")
                        st.write(f"Approved loans: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test):.1%})")
                except Exception as e:
                    st.error(f"Error splitting data: {str(e)}")
                    st.write("Please check your data for issues.")
        
        # Model training
        st.subheader("Model Training")
        
        # Prepare data for modeling - handle the case where Loan_ID might not exist
        X = st.session_state.preprocessed_df.copy()
        if 'Loan_Status' in X.columns:
            y = X['Loan_Status']
            X = X.drop('Loan_Status', axis=1)
        else:
            st.error("Error: No 'Loan_Status' column found in preprocessed data. Cannot train models.")
            return
        
        # Remove Loan_ID if it exists
        if 'Loan_ID' in X.columns:
            X = X.drop('Loan_ID', axis=1)
        
        # Check for any remaining NaN values in y and drop those rows
        valid_indices = ~y.isna()
        if not valid_indices.all():
            missing_count = (~valid_indices).sum()
            if missing_count == len(y):
                st.error(f"Error: All {missing_count} target values are missing. Cannot train a model.")
                return
            elif missing_count > 0:
                st.warning(f"Removing {missing_count} rows with missing target values")
                X = X[valid_indices]
                y = y[valid_indices]
        
        # Verify we have enough data to train
        if len(X) < 10:  # Arbitrary minimum threshold
            st.error(f"Error: Only {len(X)} samples remain after preprocessing. Need more data to train a model.")
            return

        # Feature importance analysis
        with st.expander("Feature Importance Analysis"):
            try:
                # Train a Random Forest for feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Plot feature importance
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                plt.title('Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.write("""
                **Key Insights from Feature Importance**:
                - Credit History is the most important feature for loan approval prediction
                - Applicant Income and Loan Amount are also significant factors
                """)
            except Exception as e:
                st.error(f"Error in feature importance analysis: {str(e)}")
                st.info("Please ensure your data is properly preprocessed and contains no missing values.")
                st.write("Data shape:", X.shape)
                st.write("First few rows of X:")
                st.dataframe(X.head())
                st.write("First few values of y:")
                st.write(y.head())
        
        # Model training options
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, key="training_split_slider") / 100
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Train and evaluate models
                results, best_model = train_models(X_train, X_test, y_train, y_test)
                
                # Save results to session state
                st.session_state.models_comparison = results
                st.session_state.best_model = best_model
                
                st.success("Model training completed!")
        
        # Display model comparison if available
        if st.session_state.models_comparison is not None:
            st.subheader("Model Comparison")
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Model': list(st.session_state.models_comparison.keys()),
                'Test Accuracy': [results['accuracy'] for results in st.session_state.models_comparison.values()],
                'CV Mean Accuracy': [results['cv_mean'] for results in st.session_state.models_comparison.values()],
                'CV Std Dev': [results['cv_std'] for results in st.session_state.models_comparison.values()]
            })
            
            # Display comparison table
            st.dataframe(comparison_df.sort_values('Test Accuracy', ascending=False))
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            comparison_df.sort_values('Test Accuracy').plot(x='Model', y=['Test Accuracy', 'CV Mean Accuracy'], 
                                                           kind='bar', ax=ax)
            plt.title('Model Accuracy Comparison')
            plt.ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display best model details
            best_model_name = comparison_df.sort_values('Test Accuracy', ascending=False).iloc[0]['Model']
            st.subheader(f"Best Model: {best_model_name}")
            
            # Get best model from session state
            best_model_results = st.session_state.models_comparison[best_model_name]
            best_model = best_model_results['model']
            
            # Detailed evaluation of best model
            with st.expander("Detailed Evaluation of Best Model"):
                # Prepare data for evaluation
                X = st.session_state.preprocessed_df.drop(['Loan_Status'], axis=1)
                if 'Loan_ID' in X.columns:
                    X = X.drop('Loan_ID', axis=1)
                
                # Get the target and ensure no NaN values
                y = st.session_state.preprocessed_df['Loan_Status']
                
                # Remove rows with NaN in the target variable
                valid_mask = ~y.isna()
                if not valid_mask.all():
                    nan_count = (~valid_mask).sum()
                    st.warning(f"Removing {nan_count} rows with NaN values in target variable")
                    X = X[valid_mask]
                    y = y[valid_mask]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]
                
                # Classification report - check for NaN values
                st.write("**Classification Report:**")
                try:
                    # Double check for NaN values before generating report
                    if np.isnan(y_test).any() or np.isnan(y_pred).any():
                        st.error("Cannot generate classification report: NaN values present in data")
                    else:
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                except Exception as e:
                    st.error(f"Error generating classification report: {str(e)}")
                    st.write("Summary metrics:")
                    try:
                        accuracy = (y_pred == y_test).mean()
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    except:
                        st.write("Unable to calculate metrics due to data issues")
                
                # Confusion Matrix
                st.write("**Confusion Matrix:**")
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title('Confusion Matrix')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating confusion matrix: {str(e)}")
                
                # ROC Curve - handle potential errors
                st.write("**ROC Curve:**")
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend(loc='lower right')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating ROC curve: {str(e)}")

# Prediction Page
def prediction_page():
    st.title("Loan Approval Prediction")
    
    # Check if model is trained
    if st.session_state.best_model is None:
        st.warning("Please train a model first in the Model Training page.")
        return
    
    # Create tabs for single prediction and batch prediction
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Enter Applicant Information")
        
        col1, col2 = st.columns(2)
        
        # Column 1 inputs
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
        # Column 2 inputs
        with col2:
            applicant_income = st.number_input("Applicant Income (₹)", min_value=0)
            coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
            loan_term = st.number_input("Loan Term (in months)", min_value=0)
            credit_history = st.selectbox("Credit History", [1, 0], 
                                         help="1: has all loans paid on time, 0: has delayed payments")
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
        
        # Prediction button
        if st.button("Predict Loan Status"):
            # Collect user input
            user_input = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }
            
            # Preprocess input
            processed_input = preprocess_input(user_input)
            
            # Make prediction
            prediction = st.session_state.best_model.predict(processed_input)
            probability = st.session_state.best_model.predict_proba(processed_input)
            
            # Display prediction
            st.subheader("Prediction Result")
            
            # Create columns for result and probability
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.success("✅ Loan Approved")
                else:
                    st.error("❌ Loan Rejected")
            
            with col2:
                approval_prob = probability[0][1] * 100
                st.metric("Approval Probability", f"{approval_prob:.2f}%")
            
            # Display gauge chart for probability
            fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
            approval_prob_normalized = approval_prob / 100
            bars = ax.bar([0], [1], width=0.5, bottom=0, alpha=0.2, color='lightgrey')
            bars = ax.bar([0], [approval_prob_normalized], width=0.5, bottom=0, 
                         color='green' if approval_prob_normalized >= 0.5 else 'red')
            
            # Remove grid and spines
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines.clear()
            
            # Add text in the center
            plt.text(0, 0, f"{approval_prob:.1f}%", ha='center', va='center', fontsize=20)
            
            # Set limits
            ax.set_ylim(0, 1)
            
            st.pyplot(fig)
            
            # Factors influencing decision
            if st.session_state.best_model.__class__.__name__ == 'RandomForestClassifier':
                st.subheader("Factors Influencing Decision")
                
                # Get feature importances
                feature_importance = pd.DataFrame({
                    'Feature': processed_input.columns,
                    'Value': processed_input.values[0],
                    'Importance': st.session_state.best_model.feature_importances_
                })
                
                # Sort by importance
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                # Display top factors
                st.write("Top factors influencing the prediction:")
                
                for i, row in feature_importance.head(3).iterrows():
                    st.write(f"**{row['Feature']}**: {row['Value']} (Importance: {row['Importance']:.4f})")
    
    with tab2:
        st.subheader("Batch Prediction")
        st.write("""
        Upload a CSV file with multiple applicants' information to get predictions for all of them at once.
        
        The CSV should have the same columns as the sample data (excluding Loan_Status).
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                batch_df = pd.read_csv(uploaded_file)
                
                # Display the uploaded data
                st.subheader("Uploaded Data")
                st.dataframe(batch_df.head())
                
                # Check if required columns are present
                required_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                                'Loan_Amount_Term', 'Credit_History', 'Property_Area']
                
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Process batch prediction
                    if st.button("Run Batch Prediction"):
                        with st.spinner("Processing batch predictions..."):
                            # Preprocess batch data
                            batch_processed = batch_df.copy()
                            
                            # Handle missing values
                            for col in batch_processed.columns:
                                if col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
                                    batch_processed[col].fillna(batch_processed[col].median(), inplace=True)
                                else:
                                    batch_processed[col].fillna(batch_processed[col].mode()[0], inplace=True)
                            
                            # Encode categorical features
                            categorical_cols = st.session_state.cat_columns
                            batch_processed[categorical_cols] = st.session_state.fitted_encoder.transform(batch_processed[categorical_cols])
                            
                            # Make predictions
                            batch_predictions = st.session_state.best_model.predict(batch_processed[required_cols])
                            batch_probabilities = st.session_state.best_model.predict_proba(batch_processed[required_cols])[:, 1]
                            
                            # Add predictions to the original data
                            result_df = batch_df.copy()
                            result_df['Predicted_Loan_Status'] = ['Approved' if pred == 1 else 'Rejected' for pred in batch_predictions]
                            result_df['Approval_Probability'] = batch_probabilities
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(result_df)
                            
                            # Summary statistics
                            st.subheader("Summary")
                            approval_rate = (batch_predictions == 1).mean() * 100
                            st.metric("Overall Approval Rate", f"{approval_rate:.2f}%")
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">Download Results as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Main app
def main():
    page = navigation()
    
    if page == "Data Analysis":
        data_analysis_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Prediction":
        prediction_page()

if __name__ == "__main__":
    main()
