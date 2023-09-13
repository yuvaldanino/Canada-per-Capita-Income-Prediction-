import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('canada_per_capita_income.csv')


new_df = df.drop('income',axis='columns')
income = df.income

#making linear regression object 
reg = linear_model.LinearRegression()
reg.fit(new_df, income)

# Generate a sequence of future years (e.g., 50 years)
future_years = np.arange(df['year'].max() + 1, df['year'].max() + 51)

# Predict income values for the next 50 years
future_income_predictions = reg.predict(np.expand_dims(future_years, axis=1))


# Plot the original data points
plt.scatter(df['year'], income, color='blue', label='Data Points')

# Plot the linear regression line for the original data
plt.plot(df['year'], reg.predict(new_df), color='red', linewidth=3, label='Linear Regression Line (Original Data)')

# Plot the predicted income values for the next 50 years
plt.plot(future_years, future_income_predictions, color='green', linestyle='--', label='Predicted Income (Next 50 Years)')

# Add labels and a legend
plt.xlabel('Year')
plt.ylabel('Income')
plt.legend()

plt.show()
