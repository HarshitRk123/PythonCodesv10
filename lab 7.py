import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import  ColumnTransformer
from sklearn.pipeline import Pipeline

Data=pd.read_csv('large_data.csv')

# Data = {
#     'Age': [22, 25, 47, 52, 46],
#     'Salary': [50000, 60000, 70000, 80000, 90000],
#     'Country': ['USA', 'UK', 'India', 'USA', 'UK']
# }

df = pd.DataFrame(Data)
print("Original DataFrame:")
print(df)

X = df[['Age', 'Salary', 'Country']]

preprocessor = ColumnTransformer ( transformers = [
        ('num', StandardScaler(), ['Age', 'Salary']),
        ('cat', OneHotEncoder(), ['Country'])
    ]
)

pipeline =  Pipeline(steps=[('preprocessor', preprocessor)])

X_processed = pipeline.fit_transform(X)

column_names = (pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['Age', 'Salary']).tolist() + 
                list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(['Country'])))

df_processed = pd.DataFrame(X_processed,  columns=column_names) 
print("\n Processed DataFrames:")
print(df_processed)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['Age'], df['Salary'], color='blue')
plt.title('Original Data: Age vs Salary')
plt.xlabel ('Age')
plt.ylabel ('Salary')

plt.subplot(1, 2, 2)
plt.scatter(df_processed['Age'], df_processed['Salary'], color='orange')
plt.title('Processed Data: Age vs Salary (Scaled)')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Salary')

plt.tight_layout()
plt.show()
 