import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

np.random.seed(42)  
num_samples = 1000

features = np.random.rand(num_samples, 3) * 100 
labels = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]) 

df = pd.DataFrame(features, columns=['word_freq', 'capital_run_length', 'special_char_freq'])
df['spam'] = labels  

X = df[['word_freq', 'capital_run_length', 'special_char_freq']]  
y = df['spam'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'])

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Not Spam', 'Spam'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()