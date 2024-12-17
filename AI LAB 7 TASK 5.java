import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

#Sample data: Patient symptoms and disease diagnosis
data = {
'Fever': [1, 1, 0, 1, 0, 0, 1, 0, 1, 0],
'Cough': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
'Fatigue': [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
'Diagnosis': ['Disease A', 'Disease A', 'Disease B', 'Disease A', 'Disease B', 'Disease A', 'Disease B', 'Disease B', 'Disease A', 'Disease B']
}

#Convert the data into a pandas Dataframe
df = pd.DataFrame(data)

#Features (X) and Target (y)
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

#Encoding the target variable ('Disease A' -> 0, 'Disease B' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

#Train the model
dt_classifier.fit(X_train, y_train)

#Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Classification report
print("\nClassification Report:")