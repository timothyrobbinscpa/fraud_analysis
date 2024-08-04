#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/datasets/ealaxi/banksim1/data
# 
# 

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras_tuner as kt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from keras_tuner import Objective

from sklearn.metrics import f1_score, make_scorer


start_time = time.time()

plt.style.use('ggplot')


# In[3]:


import pandas as pd

file_path = '../data/fraud_dataset.csv'

test_df = pd.read_csv(file_path)
   
test_df.head()


# In[4]:


# Remove quotes from columns

import pandas as pd

# Function to remove single quotes
def remove_quotes(x):
    return x.replace("'", "") if isinstance(x, str) else x

# Specify the columns for which you want to remove single quotes
converters = {'age': remove_quotes, 'gender': remove_quotes, 'category': remove_quotes, 'customer': remove_quotes, 'zipcodeOri': remove_quotes, 'merchant': remove_quotes, 'zipMerchant': remove_quotes}

# Import data with converters
df = pd.read_csv('../data/fraud_dataset.csv', converters=converters)

# Print the first few rows to confirm
print(df.head())


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.describe(include=['O'])


# In[9]:


df.isna().sum()


# In[10]:


df.duplicated().sum()


# # Dataset overview
# 
# We detect the fraudulent transactions from the Banksim dataset. This synthetically generated dataset consists of payments from various customers made in different time periods and with different amounts. The features are as follows:
# 
# - Step: This feature represents the day from the start of simulation. It has 180 steps so simulation ran for virtually 6 months.
# - Customer: This feature represents the customer id
# - zipCodeOrigin: The zip code of origin/source.
# - Merchant: The merchant's id
# - zipMerchant: The merchant's zip code
# - Age: Categorized age
#     - 0: <= 18,
#     - 1: 19-25,
#     - 2: 26-35,
#     - 3: 36-45,
#     - 4: 46-55,
#     - 5: 56-65,
#     - 6: > 65
#     - U: Unknown
# - Gender: Gender for customer
#     - E : Enterprise,
#     - F: Female,
#     - M: Male,
#     - U: Unknown
# - Category: Category of the purchase. I won't write all categories here, we'll see them later in the analysis.
# - Amount: Amount of the purchase
# - Fraud: Target variable which shows if the transaction fraudulent(1) or benign(0)
# 
# 

# # EDA

# In[13]:


df.head()


# The two zip code columns are constant for each value and thus provide no additional information.  These will be dropped.

# In[15]:


# Move to preprocessing

df.drop(columns=['zipcodeOri', 'zipMerchant'], inplace=True)


# In[16]:


# Adjust target variable to be categorical
#df['fraud'] = df['fraud'].astype('category') - but can't create barplots if neither variable is numeric


# ## Univariate Analysis

# ### Numerical Columns

# In[ ]:


import seaborn as sns

# Setting up the visualization environment
sns.set(style="whitegrid")

# Univariate Analysis for 'step' and 'amount'
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Histogram for 'step'
sns.histplot(data['step'], bins=30, kde=False, ax=ax[0, 0])
ax[0, 0].set_title('Transaction Frequency Over Days')
ax[0, 0].set_xlabel('Day')
ax[0, 0].set_ylabel('Frequency')

# Histogram for 'amount'
sns.histplot(data['amount'], bins=30, kde=False, ax=ax[0, 1], color='orange')
ax[0, 1].set_title('Distribution of Transaction Amounts')
ax[0, 1].set_xlabel('Amount')
ax[0, 1].set_ylabel('Frequency')

# Boxplot for 'step'
sns.boxplot(x=data['step'], ax=ax[1, 0])
ax[1, 0].set_title('Boxplot of Days')
ax[1, 0].set_xlabel('Day')

# Boxplot for 'amount'
sns.boxplot(x=data['amount'], ax=ax[1, 1], color='orange')
ax[1, 1].set_title('Boxplot of Transaction Amounts')
ax[1, 1].set_xlabel('Amount')

plt.tight_layout()
plt.show()

# Summary statistics for 'step' and 'amount'
step_summary = data['step'].describe()
amount_summary = data['amount'].describe()

step_summary, amount_summary


# **Univariate Analysis Results - Numerical Columns**
# - Step (Time):
#     - Histogram: Transactions are relatively evenly distributed over the 180 days, with a slight increase in frequency towards the end. This suggests a stable usage pattern with some growth or seasonal effects.
#     - Boxplot: Indicates that the data is uniformly distributed across the time frame, with no outliers.
#     - Summary Statistics: 
#         - Mean day is approximately 95 (middle of the 180-day period).  
#         - Standard deviation of 51 days indicates a uniform spread throughout the period.
#         
# - Amount:
#     - Histogram: The distribution of transaction amounts is heavily right-skewed, showing that most transactions are of lower value, with a few high-value outliers.
#     - Boxplot: Confirms the presence of outliers with very high transaction amounts.
#     - Summary Statistics: 
#         - Mean transaction amount is approximately $37.89.
#         - A high standard deviation ($111.40) relative to the mean suggests significant variability, primarily driven by outliers.
#         - The median ($26.90) being lower than the mean also indicates a right-skewed distribution.

# ### Categorical Columnns

# In[ ]:


# Univariate Analysis for categorical variables
fig, ax = plt.subplots(3, 2, figsize=(14, 18))

# Age distribution
sns.countplot(x='age', data=data, ax=ax[0, 0], palette='viridis')
ax[0, 0].set_title('Distribution of Age Groups')
ax[0, 0].set_xlabel('Age Group')
ax[0, 0].set_ylabel('Frequency')

# Gender distribution
sns.countplot(x='gender', data=data, ax=ax[0, 1], palette='Set2')
ax[0, 1].set_title('Distribution of Genders')
ax[0, 1].set_xlabel('Gender')
ax[0, 1].set_ylabel('Frequency')

# Category distribution
sns.countplot(x='category', data=data, ax=ax[1, 0], palette='coolwarm')
ax[1, 0].set_title('Distribution of Transaction Categories')
ax[1, 0].set_xlabel('Category')
ax[1, 0].set_ylabel('Frequency')
ax[1, 0].tick_params(axis='x', rotation=90)

# Fraud distribution
sns.countplot(x='fraud', data=data, ax=ax[1, 1], palette='Set1')
ax[1, 1].set_title('Distribution of Fraud Status')
ax[1, 1].set_xlabel('Fraud')
ax[1, 1].set_ylabel('Frequency')

# Simplifying further plots by removing 'customer' and 'merchant' due to high cardinality
# Plots for 'customer' and 'merchant' might not be meaningful due to the large number of unique values
plt.tight_layout()
plt.show()


# **Univariate Analysis Results - Categorical Columns**
# 
# - Age Group: Distribution shows that certain age groups are more prevalent in the dataset. This can help identify which groups are more active in transactions.
# - Gender: The gender distribution helps understand the demographic split of the dataset users. There's a visible distribution between different genders which may influence transaction patterns.
# - Transaction Categories: There's significant variation in the frequency of different transaction categories. Some categories are much more common, which may impact the focus of fraud detection strategies. Categories with fewer transactions but higher fraud rates could be particularly important.
# - Fraud Status: This plot highlights the imbalance between fraudulent and non-fraudulent transactions, with non-fraudulent transactions significantly outnumbering fraudulent ones. Such an imbalance is typical in fraud detection scenarios and poses challenges in modeling and analysis.

# Let's analyze the top merchants and customers in terms of transaction frequency and fraud incidence to identify any patterns or outliers. We'll focus on the top 10 merchants and customers based on the total number of transactions, and then examine their association with fraudulent transactions.
# 

# In[ ]:


# Calculating the top 10 merchants and customers by transaction volume
top_merchants = data['merchant'].value_counts().head(10)
top_customers = data['customer'].value_counts().head(10)

# Calculating fraud incidences for top 10 merchants and customers
top_merchants_fraud = data[data['fraud'] == 1]['merchant'].value_counts().reindex(top_merchants.index, fill_value=0)
top_customers_fraud = data[data['fraud'] == 1]['customer'].value_counts().reindex(top_customers.index, fill_value=0)

# Plotting results
fig, ax = plt.subplots(2, 2, figsize=(14, 12))

# Top 10 merchants by transaction volume
ax[0, 0].bar(top_merchants.index, top_merchants, color='blue')
ax[0, 0].set_title('Top 10 Merchants by Transaction Volume')
ax[0, 0].set_xlabel('Merchant ID')
ax[0, 0].set_ylabel('Number of Transactions')
ax[0, 0].tick_params(axis='x', rotation=45)

# Fraud transactions for top 10 merchants
ax[0, 1].bar(top_merchants_fraud.index, top_merchants_fraud, color='red')
ax[0, 1].set_title('Fraud Transactions for Top 10 Merchants')
ax[0, 1].set_xlabel('Merchant ID')
ax[0, 1].set_ylabel('Number of Fraudulent Transactions')
ax[0, 1].tick_params(axis='x', rotation=45)

# Top 10 customers by transaction volume
ax[1, 0].bar(top_customers.index, top_customers, color='green')
ax[1, 0].set_title('Top 10 Customers by Transaction Volume')
ax[1, 0].set_xlabel('Customer ID')
ax[1, 0].set_ylabel('Number of Transactions')
ax[1, 0].tick_params(axis='x', rotation=45)

# Fraud transactions for top 10 customers
ax[1, 1].bar(top_customers_fraud.index, top_customers_fraud, color='orange')
ax[1, 1].set_title('Fraud Transactions for Top 10 Customers')
ax[1, 1].set_xlabel('Customer ID')
ax[1, 1].set_ylabel('Number of Fraudulent Transactions')
ax[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Top 10 Merchants:
# - Transaction Volume: The bar graph shows that some merchants have significantly higher transaction volumes than others.
# - Fraud Incidences: For these top merchants, the number of fraudulent transactions varies. Some merchants have a relatively higher incidence of fraud, which could suggest areas where increased monitoring might be necessary.
# 
# Top 10 Customers:
# - Transaction Volume: Similar to merchants, some customers engage in far more transactions than others, highlighting their potential influence or importance in the dataset.
# - Fraud Incidences: The distribution of fraudulent transactions among these customers shows that fraud is not uniformly spread, indicating that certain customers might be more susceptible to fraud or involved in fraudulent activities.

# Explore the zipcodeOri and zipMerchant to confirm if any variability exists.

# In[ ]:





# ## Bivariate Analysis

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# As would be expected, fraud tends to occur at larger amounts.

# Fraud tends to occur nearly twice as frequently with individuals 18 or under. 

# Most fraudalent cases occur with leisure and travel followed by sports and toys. 

# In[ ]:





# In[ ]:





# In[ ]:





# # Model Preparation and Modeling

# In[50]:


import shutil

# Directory where the tuner data is stored
#tuner_directory = 'C:/Users/trobb/GitHub/Projects/timothyrobbinscpa/fraud_analysis/src/my_dir/keras_tune_nn'
tuner_directory = 'C:/Users/trobb/GitHub/Projects/timothyrobbinscpa/fraud_analysis/src/hyperparam_tuning/neural_network_tuning'

# Remove the directory to reset the tuner
shutil.rmtree(tuner_directory, ignore_errors=True)



# ## Neural Network Model

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, Objective, RandomSearch


# In[52]:


class F1Score(tf.keras.metrics.Metric):
    """
    Custom implementation of the F1 Score as a Keras metric. The F1 Score is the harmonic mean of precision and recall,
    providing a balance between the two metrics. It is particularly useful when dealing with imbalanced datasets.
    
    Attributes:
        true_positives (tf.Variable): Tracks the number of true positives encountered over all batches.
        false_positives (tf.Variable): Tracks the number of false positives encountered over all batches.
        false_negatives (tf.Variable): Tracks the number of false negatives encountered over all batches.
    """

    def __init__(self, name='f1_score', **kwargs):
        """
        Initializes the F1Score metric instance.
        
        Args:
            name (str, optional): Name of the metric instance, defaults to 'f1_score'.
            **kwargs: Arbitrary keyword arguments, passed to the parent class.
        """
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric, accumulating the true positives, false positives, and false negatives 
        predictions based on the input labels and predictions.
        
        Args:
            y_true (tf.Tensor): The ground truth labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): Optional weighting of each example, defaults to None.
        """
        y_pred = tf.round(y_pred)  # Convert probabilities to binary values (0 or 1)
        y_true = tf.cast(y_true, tf.float32)  # Cast y_true to float32 to match y_pred data type
        tp = tf.reduce_sum(y_true * y_pred)  # Calculate true positives
        fp = tf.reduce_sum(y_pred) - tp  # Calculate false positives
        fn = tf.reduce_sum(y_true) - tp  # Calculate false negatives

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """
        Computes and returns the F1 Score using the current state of the metric.
        
        Returns:
            tf.Tensor: The computed F1 Score.
        """
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        """
        Resets all of the metric state variables to zero.
        This function is called at the start of each epoch.
        """
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)



# In[54]:


# Separating the target variable 'fraud' from the feature set.
y = df['fraud']  # Target variable for prediction.
X = df.drop('fraud', axis=1)  # Feature set excluding the target.

# Identifying types of columns for preprocessing:
# - Categorical columns are identified by their data type ('object' or 'category').
# - Numerical columns are identified by data types 'int64' or 'float64', excluding the 'fraud' column.
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns.tolist() if col != 'fraud']


# In[56]:


# Configure preprocessing for the neural network. This includes:
# - Standardizing numerical columns to have zero mean and unit variance.
# - One-hot encoding categorical columns to convert them into a format suitable for neural network training.
# Unspecified columns are dropped to ensure the model only trains on relevant features.
nn_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Normalize numerical data.
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)  # Encode categorical data.
    ],
    remainder='drop'  # Drop other columns not specified explicitly.
)

# Apply the preprocessing transformations to the data, preparing it for the neural network.
X_nn_processed = nn_preprocessor.fit_transform(X)


# In[58]:


# Preprocessing for Random Forest:
# Label encoding is applied to categorical columns to transform them into numerical values that the model can interpret.
X_rf = X.copy()  # Create a copy of the original feature set to modify for the Random Forest model.
for col in categorical_cols:
    le = LabelEncoder()
    X_rf[col] = le.fit_transform(X_rf[col])  # Encode each categorical column with label encoding.

# Apply standard scaling to numerical features to normalize their values, which helps in stabilizing the model training.
scaler = StandardScaler()
X_rf[numerical_cols] = scaler.fit_transform(X_rf[numerical_cols])  # Scale each numerical column.



# In[73]:


# Split the preprocessed data into training and testing sets for neural network and random forest models.
# Both splits use a test size of 20% and a random state for reproducibility.
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn_processed, y, test_size=0.2, random_state=42)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y, test_size=0.2, random_state=42)

# SMOTE (Synthetic Minority Over-sampling Technique) is applied to address class imbalance by generating synthetic samples.
smote = SMOTE(random_state=42)
X_train_nn_smote, y_train_nn_smote = smote.fit_resample(X_train_nn, y_train_nn)
X_train_rf_smote, y_train_rf_smote = smote.fit_resample(X_train_rf, y_train_rf)


# In[ ]:





# In[ ]:





# In[75]:


# MODELING STARTS HERE


# In[76]:


class MyHyperModel(HyperModel):
    """
    A custom hypermodel class using Keras Tuner for optimizing a neural network model for binary classification.
    
    Attributes:
        input_shape (int): The shape of the input data that the model will accept.
    """

    def __init__(self, input_shape):
        """
        Initializes the hypermodel with the required input shape for the neural network.
        
        Args:
            input_shape (int): The number of features in the input dataset.
        """
        self.input_shape = input_shape

    def build(self, hp):
        """
        Builds and compiles a neural network model with hyperparameters that can be tuned.
        
        Args:
            hp (HyperParameters): A set of hyperparameters provided by Keras Tuner to optimize the model.
        
        Returns:
            model (tf.keras.Model): The compiled neural network model.
        """
        model = Sequential([
            Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu', input_shape=(self.input_shape,)),
            Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(units=hp.Int('units_2', min_value=64, max_value=256, step=32), activation='relu'),
            Dropout(hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=[F1Score(), Precision(name='precision'), Recall(name='recall')]
        )
        return model

# Instantiate the hypermodel for the neural network.
hypermodel = MyHyperModel(input_shape=X_train_nn_smote.shape[1])


# In[122]:


# Setup the Keras Tuner for hyperparameter optimization using the RandomSearch algorithm.
# The goal is to maximize the validation F1 score, limiting the search to a single trial for quick testing.
tuner = RandomSearch(
    hypermodel,
    objective=Objective('val_f1_score', direction='max'),  # Maximize F1 score
    max_trials=3,  # Run 3 sets of hyperparameter values
    executions_per_trial=1,  # Execute the model once per trial
    directory='hyperparam_tuning',
    project_name='neural_network_tuning'
)

# Configure early stopping to prevent overfitting by monitoring the validation F1 score.
early_stopper = EarlyStopping(
    monitor='val_f1_score',
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restores model weights from the epoch with the best value of the monitored quantity
    mode='max'  # The monitoring metric is expected to be maximized
)

# Perform the hyperparameter search over the training data with a validation split.
tuner.search(
    X_train_nn_smote, y_train_nn_smote,
    epochs=20,  # Limited number of epochs for quick iteration
    validation_split=0.2,  # 20% of the training data is used as validation data
    callbacks=[early_stopper]
)

# Retrieve the best model from the tuning session.
best_model = tuner.get_best_models(num_models=1)[0]

# Fit the best model on the training data with validation using the test set.
history = best_model.fit(
    X_train_nn_smote, y_train_nn_smote,
    epochs=20,  # Number of epochs to train the model
    validation_data=(X_test_nn, y_test_nn),
    callbacks=[early_stopper]
)

# Evaluate the neural network model on the test data.
nn_predictions = (best_model.predict(X_test_nn) > 0.5).astype(int)  # Binary classification threshold
nn_prob_predictions = best_model.predict(X_test_nn).ravel()  # Probability predictions

# Print classification metrics and ROC-AUC score for model evaluation.
print("Neural Network Classification Report:\n", classification_report(y_test_nn, nn_predictions))
print("Neural Network ROC-AUC Score:", roc_auc_score(y_test_nn, nn_prob_predictions))


# In[124]:


import matplotlib.pyplot as plt

# Adjust the figure size to better accommodate two plots side by side
plt.figure(figsize=(12, 5))  # Wider figure to prevent squeezing of plots

# Plot for training and validation loss
plt.subplot(1, 2, 1)  # One row, two columns, first plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot for F1 Score during training
plt.subplot(1, 2, 2)  # One row, two columns, second plot
plt.plot(history.history['f1_score'], label='Train F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.title('F1 Score During Training')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[ ]:





# In[ ]:





# In[81]:


from sklearn.metrics import f1_score, make_scorer

# Create a custom scorer object that measures the F1 score for RF Model:
# The `make_scorer` function converts the `f1_score` metric into a scorer that can be used with scikit-learn's model evaluation and parameter tuning tools.
# The F1 score is a harmonic mean of precision and recall and is particularly useful for evaluating models on imbalanced datasets.
f1_scorer = make_scorer(f1_score)


# In[137]:


# Initialize Random Forest classifier.
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid that RandomizedSearchCV will explore during tuning.
param_grid = {
    'n_estimators': randint(100, 300),  # Number of trees in the forest: increases model complexity and accuracy.
    'max_features': ['sqrt', 'log2'],   # Number of features to consider for the best split: helps in reducing overfitting.
    'max_depth': randint(10, 40),       # Maximum depth of each tree: controls overfitting by limiting how deep trees can grow.
    'min_samples_split': randint(2, 10),# Minimum number of samples required to split an internal node: higher numbers reduce model complexity.
    'min_samples_leaf': randint(1, 4),  # Minimum number of samples required at a leaf node: prevents overfitting on very small leaf sizes.
    'bootstrap': [True, False],          # Method for sampling data points (with or without replacement).
    'class_weight': ['balanced', 'balanced_subsample']  # Handling class imbalance.

}


# Setup RandomizedSearchCV with the defined grid and a focus on maximizing the F1 score.
rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,  # Number of parameter settings sampled.
    cv=3,      # Number of folds in cross-validator.
    scoring=f1_scorer,  # Custom F1 scorer defined earlier to optimize for F1 score.
    verbose=1, # Controls the verbosity: the higher, the more messages about the process.
    random_state=42,
    n_jobs=-1  # Use all available cores to perform the computations.
)

# Fit the RandomizedSearchCV to the SMOTE-enhanced training data to find the best parameters.
rf_random_search.fit(X_train_rf_smote, y_train_rf_smote)

# Retrieve the best estimator from the search.
best_rf = rf_random_search.best_estimator_

# Make predictions using the best Random Forest model found.
rf_predictions = best_rf.predict(X_test_rf)
rf_prob_predictions = best_rf.predict_proba(X_test_rf)[:, 1]

# Output the classification report and ROC-AUC score for the Random Forest model.
print("Random Forest Classification Report:\n", classification_report(y_test_rf, rf_predictions))
print("Random Forest ROC-AUC Score:", roc_auc_score(y_test_rf, rf_prob_predictions))


# In[140]:


# After fitting the Random Forest model:
rf_feature_importances = best_rf.feature_importances_


# In[141]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Define the performance metrics for comparison.
metrics = ['Precision', 'Recall', 'F1 Score', 'ROC AUC']

# Compute scores for the Neural Network model.
nn_scores = [
    precision_recall_fscore_support(y_test_nn, nn_predictions, average='binary')[0],  # Precision
    precision_recall_fscore_support(y_test_nn, nn_predictions, average='binary')[1],  # Recall
    precision_recall_fscore_support(y_test_nn, nn_predictions, average='binary')[2],  # F1 Score
    roc_auc_score(y_test_nn, nn_prob_predictions)  # ROC AUC
]

# Compute scores for the Random Forest model.
rf_scores = [
    precision_recall_fscore_support(y_test_rf, rf_predictions, average='binary')[0],  # Precision
    precision_recall_fscore_support(y_test_rf, rf_predictions, average='binary')[1],  # Recall
    precision_recall_fscore_support(y_test_rf, rf_predictions, average='binary')[2],  # F1 Score
    roc_auc_score(y_test_rf, rf_prob_predictions)  # ROC AUC
]

# Setup for plotting the comparison.
x = np.arange(len(metrics))  # Label locations for metrics
width = 0.35  # Width of the bars

# Create a figure and a set of subplots.
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, nn_scores, width, label='Neural Network')  # Bars for Neural Network
rects2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest')   # Bars for Random Forest

# Add some text for labels, title, and custom x-axis tick labels.
ax.set_ylabel('Scores')
ax.set_title('Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add text annotations above each bar for clarity.
def add_value_labels(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels(ax, rects1)
add_value_labels(ax, rects2)

# Improve the layout to make room for the tick labels.
fig.tight_layout()

# Display the plot.
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# The confusion matrices for the two models indicate the following:
# 
# - The Random Forest has a lower False Negative rate, indicating it is better at detecting fraud (higher recall). However, it has a higher False Positive rate, which would result in more legitimate transactions being flagged as fraud.
# - The Neural Network has a lower False Positive rate, leading to fewer legitimate transactions being incorrectly flagged (higher precision). But it has a higher False Negative rate, meaning it misses more actual fraud cases.
# 

# Accuracy:
# 
# Both models perform similarly, with very high accuracy scores: 1.0 for the Random Forest and 0.99 for the Neural Network.
# Accuracy measures the proportion of true results (both true positives and true negatives) among the total number of cases examined.
# Precision:
# 
# The Random Forest model has a recall of 0.77, whereas the Neural Network has a recall of 0.84.
# Precision measures the proportion of true positive identifications, meaning it is the number of true positives divided by the number of true positives plus the number of false positives. A higher precision score indicates a lower false positive rate.
# Recall:
# 
# The Random Forest model has a precision of 0.9, which is significantly higher than the Neural Network's precision of 0.48.
# Recall (also known as sensitivity) measures the proportion of actual positives that were identified correctly. It is the number of true positives divided by the number of true positives plus the number of false negatives. A higher recall score indicates that the model is better at capturing the positive class.
# F1 Score:
# 
# The Random Forest model has an F1 score of 0.8, which is higher than the Neural Network's F1 score of 0.62.
# The F1 Score is the harmonic mean of Precision and Recall. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a model has a good balance between precision and recall.
# Overall Interpretation:
# The Random Forest model appears to have better recall and F1 scores, indicating it is better at correctly identifying the positive class and balancing precision and recall.
# The Neural Network has a slightly better precision but a significantly lower recall, suggesting it is more conservative in predicting the positive class but misses a substantial number of positive cases.
# In terms of accuracy, both models perform exceptionally well, but this metric alone can be misleading, especially if the class distribution is imbalanced. It's important to consider the context of the problem: if false positives and false negatives have different costs, precision and recall might be more important than accuracy.
# When deciding between the two models, you would consider the specific requirements of your application. If minimizing false negatives is crucial (thus requiring a high recall), the Random Forest seems preferable. If you want to ensure that the positive predictions are correct (requiring high precision), then the Neural Network might be the better choice, albeit only slightly.
# 
# Lastly, the F1 score suggests that the Random Forest has a better overall balance of precision and recall, making it a good choice if you value both metrics equally.
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ### PR ROC Curve

# In[142]:


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Assuming 'nn_prob_predictions' are the prediction probabilities for the positive class from the Neural Network model
# and 'rf_prob_predictions' are the prediction probabilities for the positive class from the Random Forest model.
# Also assuming 'y_test_nn' is the actual labels for the test dataset used for both models.

# Calculate the precision-recall curve and AUC for the Neural Network
nn_precision, nn_recall, _ = precision_recall_curve(y_test_nn, nn_prob_predictions)
nn_auc = auc(nn_recall, nn_precision)

# Calculate the precision-recall curve and AUC for the Random Forest
rf_precision, rf_recall, _ = precision_recall_curve(y_test_rf, rf_prob_predictions)
rf_auc = auc(rf_recall, rf_precision)

# Plotting the Precision-Recall curves
plt.figure(figsize=(8, 6))
plt.plot(nn_recall, nn_precision, label=f'Neural Network PR (AUC = {nn_auc:.4f})', linestyle='-', linewidth=2)
plt.plot(rf_recall, rf_precision, label=f'Random Forest PR (AUC = {rf_auc:.4f})', linestyle='--', linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# a Precision-Recall (PR) Curve, which is used to evaluate the performance of a classification model at different probability thresholds. The PR curve plots the Precision (y-axis) against the Recall (x-axis).
# 
# Here’s an interpretation of the plot:
# 
# Precision-Recall Trade-off: The curves show the trade-off between precision and recall for the Random Forest and Neural Network models. As the recall increases, precision tends to decrease, which is typical behavior for classifiers.
# 
# Random Forest Model (Red Curve):
# 
# The Random Forest model has a Precision-Recall Area Under the Curve (PR AUC) of 0.8960.
# It starts with high precision but experiences a gradual decline as recall increases.
# The relatively high PR AUC value suggests that the Random Forest model has a good balance between precision and recall overall.
# Neural Network Model (Blue Curve):
# 
# The Neural Network model has a PR AUC of 0.8486, which is slightly lower than the Random Forest model.
# This curve also demonstrates the trade-off, with precision decreasing as recall increases.
# While the Neural Network performs well, it is slightly outperformed by the Random Forest model in terms of the area under the PR curve.
# Comparison:
# 
# Both models perform relatively well, with PR AUC scores close to 1, indicating strong performance.
# However, the Random Forest model has a higher PR AUC, suggesting it is better at distinguishing between the positive and negative classes across different thresholds.
# The curves are close to each other, which means the performance of both models is somewhat similar, but the Random Forest has a slight edge.
# PR AUC Scores:
# 
# The PR AUC scores are included in the legend. Higher PR AUC scores are generally better, indicating a model that maintains high precision across different levels of recall.
# These scores provide a single measure of performance across all classification thresholds, unlike the curves that show performance at each threshold.
# In conclusion, based on this PR curve, the Random Forest model is slightly superior to the Neural Network for the task at hand, likely due to its ability to maintain higher precision at similar levels of recall. When choosing between models, if precision is more critical for the application (i.e., minimizing false positives), the Random Forest might be the preferred model. If you need to prioritize recall (i.e., minimizing false negatives), the performance difference between the models is less pronounced.
# 
# 
# 
# 

# ### Feature Importances

# In[126]:


'''

# After fitting the ColumnTransformer to your data:
nn_feature_names = []

# Iterate through transformers in the ColumnTransformer
for transformer in nn_preprocessor.transformers_:
    trans_name, trans, column_names = transformer
    
    # Handling different transformer outputs
    if hasattr(trans, 'get_feature_names_out'):  # For transformers with this method (e.g., OneHotEncoder)
        if isinstance(column_names, str):  # This means all columns are used in this transformer
            trans_feature_names = trans.get_feature_names_out()
        else:
            trans_feature_names = trans.get_feature_names_out(column_names)
    else:
        trans_feature_names = column_names  # For transformers that do not change feature names (e.g., StandardScaler)
    
    nn_feature_names.extend(trans_feature_names)

# Now nn_feature_names contains all the modified names of features processed through ColumnTransformer

'''



# In[144]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming 'rf_feature_importances' is defined as shown above
# Assuming 'nn_feature_importances' is also defined (if applicable, otherwise, this part needs adjustment based on your neural network analysis)
# Ensure the feature names are defined or extracted properly
n_features = len(rf_feature_names)  # Should be the same length as the number of importances
x = np.arange(n_features)

width = 0.35

fig, ax = plt.subplots(figsize=(15, 6))
rf_bars = ax.bar(x - width/2, rf_feature_importances, width, label='Random Forest', color='skyblue')
nn_bars = ax.bar(x + width/2, nn_feature_importances, width, label='Neural Network', color='lightgreen')

ax.set_ylabel('Feature Importance Score')
ax.set_title('Feature Importances by Model')
ax.set_xticks(x)
ax.set_xticklabels(rf_feature_names, rotation=45)  # Assuming rf_feature_names are correctly assigned
ax.legend()

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rf_bars)
add_labels(nn_bars)

fig.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Random Forest Model (Red Bars):
# 
# The feature amount has the highest importance, indicating it is the most influential feature for the Random Forest model when making predictions.
# The category feature also appears to be quite significant, followed by merchant, suggesting these features play a strong role in the model's decisions.
# Customer has a moderate level of importance.
# Features such as step, age, and gender have very low importance in the Random Forest model, implying they have minimal impact on the model's predictions.
# Neural Network Model (Blue Bars):
# 
# Similar to the Random Forest model, the amount feature is highly important in the Neural Network model but to a slightly lesser degree.
# The category and merchant features have a noticeable level of importance, but merchant is less important compared to the Random Forest model.
# The customer feature is more important in the Neural Network model than in the Random Forest model.
# Similar to the Random Forest, step, age, and gender have lower importance, with gender having the least importance among all features.
# Comparative Insights:
# Both models agree on the high importance of the amount feature, which likely has a strong correlation with the target variable.
# The category feature is also considered important by both models, which might suggest that certain categories have a higher likelihood of being associated with the target variable.
# There is a notable difference in how the models weigh the importance of customer and merchant. The Neural Network gives more importance to customer than the Random Forest model, which might be due to differences in how these models capture and use information from the data.
# The step, age, and gender features are consistently given low importance by both models, indicating these features may not be very useful in predicting the outcome.
# Differences in feature importance between the two models can arise due to the intrinsic differences in how Random Forests and Neural Networks learn from the data. Random Forests use decision trees that make splits based on reducing entropy or Gini impurity, while Neural Networks learn a set of weights through backpropagation, which can capture more complex and non-linear relationships.

# In[ ]:


end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime of the program is {total_time/(60 * 60)} hours")


# In[ ]:




