import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the JSON file
data_path = r'D:\MyProject\Recsify_Assignment\loan_approval_dataset.json'



# Load the JSON data
data_raw = pd.read_json(data_path, lines=True)

# Initialize a DataFrame to store the reshaped data
data_reshaped = pd.DataFrame()

# Extract and flatten the nested dictionary columns
for column in data_raw.columns:
    nested_dict = data_raw[column][0]
    flattened_data = pd.DataFrame.from_dict(nested_dict, orient='index').reset_index(drop=True)
    flattened_data.columns = [column]
    data_reshaped = pd.concat([data_reshaped, flattened_data], axis=1)

# Preprocess the data
numerical_features = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
categorical_features = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = data_reshaped.drop(columns=['Risk_Flag'])
y = data_reshaped['Risk_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(conf_matrix)
print(class_report)

feature_importances = model.named_steps['classifier'].feature_importances_
features = numerical_features + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features))
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select top N important features
N = 20  # Adjust this number as needed
top_features = importance_df.head(N)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()

# Data Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(data_reshaped['Risk_Flag'])
plt.title('Distribution of Risk_Flag')
plt.tight_layout()
plt.savefig('dist_risk_flag.png')
plt.show()

corr_matrix = data_reshaped[numerical_features + ['Risk_Flag']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('corr_matrix.png')
plt.show()

sns.pairplot(data_reshaped[numerical_features + ['Risk_Flag']])
plt.tight_layout()
plt.savefig('pairplot.png')
plt.show()




