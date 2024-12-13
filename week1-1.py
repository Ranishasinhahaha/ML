import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Initial Dataset with Missing Values:")
df.loc[5:10, 'sepal length (cm)'] = np.nan  
print(df.head(15))

df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean(), inplace=True)
print("\nDataset after Handling Missing Values:")
print(df.head(15))

encoder = OneHotEncoder(sparse_output=False)
encoded_target = encoder.fit_transform(df[['target']])
encoded_df = pd.DataFrame(encoded_target, columns=encoder.get_feature_names_out(['target']))
df = pd.concat([df, encoded_df], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(['target'], axis=1))  
scaled_df = pd.DataFrame(scaled_features, columns=df.drop(['target'], axis=1).columns)

scaled_df['target'] = df['target']
print("\nDataset after Feature Scaling:")
print(scaled_df.head(15))

plt.figure(figsize=(8, 6))
plt.hist(df['sepal length (cm)'], bins=15, edgecolor='black', color='skyblue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='target', palette='Set1')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

correlation_matrix = df.drop('target', axis=1).corr()  
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

X = df.drop(['target'], axis=1)  
y = df['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Features (X_train):")
print(X_train.head())
print("\nTesting Features (X_test):")
print(X_test.head())