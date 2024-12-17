import pandas as pd
import numpy as np

# Sample dictionary with students' scores, some of which are missing (NaN)
data = {
    'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Math': [90, 85, np.nan, 92, np.nan],
    'Science': [88, np.nan, 80, np.nan, 91],
    'English': [np.nan, 78, 85, 87, 90]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Print the DataFrame before filling missing values
print("Before Forward Fill:")
print(df)

# Perform forward fill to fill missing values
df_filled = df.fillna(method='ffill')

# Print the DataFrame after forward fill
print("\nAfter Forward Fill:")
print(df_filled)
