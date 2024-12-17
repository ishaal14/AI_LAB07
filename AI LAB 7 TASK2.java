import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset (You can replace this with actual data)
# Suppose the columns are: 'Math', 'Science', 'Attendance', 'Previous_GPA', and 'Performance'
data = {
    'Math': [88, 92, 78, 85, 95, 82, 91, 76, 80, 93],
    'Science': [92, 89, 75, 83, 94, 90, 82, 77, 67, 95],
    'Attendance': [1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
    'Previous_GPA': [3.5, 3.8, 3.2, 3.4, 3.9, 3.1, 3.7, 2.8, 2.5, 3.6],
    'Performance': [1, 1, 0, 1, 1, 0, 0, 1, 0, 1]  # Target variable: 1 = Good, 0 = Poor
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df.drop(columns=['Performance'])
y = df['Performance']


