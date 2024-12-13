import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

np.random.seed(42)
num_samples = 1000

data = {
    'tenure': np.random.randint(1, 60, num_samples),  # Months with the service
    'monthly_charges': np.random.uniform(20, 120, num_samples),  # Monthly payment
    'total_charges': np.random.uniform(100, 5000, num_samples),  # Total payment
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples),  # Internet type
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples),  # Contract
    'churn': np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # Churn (0 = No, 1 = Yes)
}
df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=['internet_service', 'contract_type'], drop_first=True)

X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Retained', 'Churned'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()