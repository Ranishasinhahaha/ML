import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

california_housing = fetch_california_housing()
data = california_housing['data']
target = california_housing['target']
feature_names = california_housing['feature_names']

df = pd.DataFrame(data, columns=feature_names)
df['House Price'] = target

X = df[['AveRooms', 'AveOccup']]  
y = df['House Price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (r2_score): {r2}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['AveRooms'], y=y_test, color='blue', label='True values')
sns.lineplot(x=X_test['AveRooms'], y=y_pred, color='red', label='Regression Line')
plt.title('Simple Linear Regression - House Price Prediction')
plt.xlabel('Average Number of Rooms (AveRooms)')
plt.ylabel('House Price')
plt.legend()
plt.show()