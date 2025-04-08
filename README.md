# Bank Customer Churn Prediction

## Project Overview
This project develops a neural network model to predict customer churn for a bank. Using a dataset with customer demographics and banking behavior, the model aims to identify customers who are likely to leave the bank (churn).

## Dataset Description
The dataset "Churn_Modelling.csv" contains 10,000 records with the following features:
- **RowNumber**: Row identifier
- **CustomerId**: Unique customer identifier
- **Surname**: Customer's surname
- **CreditScore**: Customer's credit score
- **Geography**: Customer's country (France, Spain, Germany)
- **Gender**: Customer's gender (Male, Female)
- **Age**: Customer's age
- **Tenure**: Number of years the customer has been with the bank
- **Balance**: Customer's account balance
- **NumOfProducts**: Number of bank products the customer uses
- **HasCrCard**: Whether the customer has a credit card (1=Yes, 0=No)
- **IsActiveMember**: Whether the customer is an active member (1=Yes, 0=No)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Whether the customer left the bank (1=Yes, 0=No) - Target variable

## Data Preprocessing Steps
1. Imported necessary libraries (pandas, numpy, sklearn, tensorflow/keras)
2. Loaded and examined the dataset (shape, duplicates, missing values)
3. Removed irrelevant columns (RowNumber, CustomerId, Surname)
4. Converted categorical variables (Geography, Gender) to dummy variables
5. Split the data into training (80%) and testing (20%) sets
6. Standardized numerical features using StandardScaler

## Model Architecture
The neural network model consists of:
- Input layer with 11 nodes (representing the 11 features)
- First hidden layer with 11 nodes and ReLU activation
- Second hidden layer with 11 nodes and ReLU activation
- Output layer with 1 node and sigmoid activation for binary classification

## Model Training
- Compiled with binary cross-entropy loss and Adam optimizer
- Trained for 100 epochs with a validation split of 20%
- Monitored training and validation loss/accuracy during training

## Model Evaluation
The final model achieved:
- Training accuracy: ~86.6%
- Validation accuracy: ~86.0% 
- Test accuracy: 86.3%

## Dependencies
- Python 3.11
- pandas
- numpy
- scikit-learn
- tensorflow 2.18.0
- matplotlib

## How to Run the Code
1. Make sure you have Python 3.11 installed on your system

2. Install the required dependencies:
   ```
   pip install pandas numpy scikit-learn tensorflow==2.18.0 matplotlib
   ```

3. Download the "Churn_Modelling.csv" dataset and place it in your working directory

4. Open the main.ipynb file in Jupyter Notebook or Google Colab

5. Run the notebook cell by cell in sequential order
   
6. Analyze the results and model performance using the generated plots and metrics
## Future Improvements
1. Hyperparameter tuning (number of layers, neurons, learning rate)
2. Feature engineering and selection
3. Try different model architectures
4. Address class imbalance (churn rate is ~20%)
5. Implement cross-validation
6. Explore other metrics (precision, recall, F1-score, ROC-AUC)
