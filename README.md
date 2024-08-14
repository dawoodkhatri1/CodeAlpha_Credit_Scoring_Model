# CodeAlpha_Credit_Scoring_Model

You can run the code in Pycharm and vscode.

I divided the task into following things:


**Importing libraries:**

pandas: Used for data manipulation and analysis.

numpy: Provides support for large arrays and matrices.

matplotlib.pyplot: Used for plotting and visualization.

sklearn.model_selection: Includes tools for splitting the dataset and performing grid search.

sklearn.pipeline: Helps in creating a machine learning pipeline.

sklearn.compose: Allows combining different preprocessing steps.

sklearn.preprocessing: Contains various preprocessing utilities.

sklearn.linear_model: Provides regression algorithms like Ridge regression.

sklearn.metrics: Offers metrics for model evaluation.

Faker: Generates fake data.


**Initialize Faker and Data Generation:**

Faker is initialized to create realistic synthetic data.

np.random.seed(0) ensures reproducibility of random numbers.

fake_data dictionary contains:
> income: Normally distributed incomes with mean 50000 and standard deviation 15000.
>> age: Random integers between 20 and 70.
>>> credit_history: Random integers between 1 and 10.
>>>> credit_score: Random integers between 300 and 850.


**Data Conversion:**

> The generated data is converted to a pandas DataFrame for easier manipulation.


**Splitting Data:**

X contains the features (income, age, credit_history).

y contains the target variable (credit_score).


**Pipline Preprocessing:**

ColumnTransformer applies transformations to specified columns.

StandardScaler standardizes features by removing the mean and scaling to unit variance.


**Pipline Regression:**

Pipeline chains preprocessing and model fitting steps.

Ridge regression is used as the model.


**HyperParameter Tuning:**

GridSearchCV performs hyperparameter tuning using cross-validation.

parameters dictionary specifies the range of alpha values for Ridge regression.


**Train-Test Split and Prediction:**

train_test_split splits the data into training and testing sets (80% train, 20% test).


**Model Evaluation:**

r2_score and mean_squared_error evaluate the model's performance.

The best hyperparameters are printed.


**Prediction Function:**

predict_credit_score: Function to predict credit scores using the trained model.

A sample prediction is made using specified income, age, and credit_history.


**Visualization:**  

plt.scatter creates a scatter plot to visualize the relationship between actual and predicted credit scores.

The output looks like this:

![image](https://github.com/user-attachments/assets/e36f034f-b53a-4b0f-89c8-92841a36812f)

![image](https://github.com/user-attachments/assets/29d9fe27-8538-49b4-a42d-dd0ba69376c0)

## License

[MIT License](LICENSE)
