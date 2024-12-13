import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42) 
study_hours = np.random.uniform(1, 10, 100)  
attendance = np.random.uniform(50, 100, 100) 
assignment_scores = np.random.uniform(60, 100, 100)  
student_performance = study_hours * 3 + attendance * 0.2 + assignment_scores * 0.5 + np.random.normal(0, 5, 100)  

df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance': attendance,
    'assignment_scores': assignment_scores,
    'student_performance': student_performance
})

print("\nMissing Values:")
print(df.isnull().sum())


X = df[['study_hours', 'attendance', 'assignment_scores']] 
y = df['student_performance']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)  # Identity line
plt.title('Actual vs Predicted Student Performance')
plt.xlabel('Actual Student Performance')
plt.ylabel('Predicted Student Performance')
plt.show()