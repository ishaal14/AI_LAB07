import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
#Sample data: Email content and whether it's spam or not (1 = Spam, 0 = Not Spam)
data = {
'email': [
"Get free money now",
"Hi, how are you?",
"Claim your free prize",
"This is your weekly update",
"Limited time offer, act now",
"Meeting at 10 AM",
"Free vacation, click now",
"Here's the report you asked for",
"Congratulations, you won a gift card",
"Important project updates"
],
'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] #1 = Spam, 0 = Not Spam
}
#Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
#Text vectorization (Bag-of-words approach)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email']).toarray() # Convert emails to feature vectors
#Target variable: 'label' (1 for Spam, 0 for Not Spam)
y = df['label']
#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Apply Linear Regression (not ideal for classification, but we proceed for demonstration)
model = LinearRegression()
#Train the model
model.fit(X_train, y_train)
#Make predictions on the test set
y_pred = model.predict(X_test)
#Apply a threshold of 0.5 to convert continuous predictions to binary labels (0 or 1)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
#Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
mse = mean_squared_error(y