# Import required packages
import pandas as pd
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'movie_metadata.csv'  # Ensure this file is in the same directory
if not os.path.exists(file_path):
    print(f"Error: Could not find data file at {file_path}")
    exit(1)

df = pd.read_csv(file_path, encoding='latin-1')

# Drop missing values for budget and imdb_score
df_budget_imdb = df[['budget', 'imdb_score']].dropna()

# Convert columns to numeric if necessary
df_budget_imdb['budget'] = pd.to_numeric(df_budget_imdb['budget'], errors='coerce')
df_budget_imdb['imdb_score'] = pd.to_numeric(df_budget_imdb['imdb_score'], errors='coerce')

# Re-drop NAs after type conversion
df_budget_imdb = df_budget_imdb.dropna()

# Add a constant to the model (intercept)
X = sm.add_constant(df_budget_imdb['budget'])
y = df_budget_imdb['imdb_score']

# Create the regression model
model = sm.OLS(y, X).fit()

# Get the coefficients
b0 = model.params[0]  # Intercept (B0)
b1 = model.params[1]  # Slope (B1)

# Print the regression equation
print(f"Regression Equation: IMDB Score = {b0:.4f} + {b1:.10f} × Budget")

# Get R-squared
r_squared = model.rsquared
print(f"R-squared: {r_squared:.4f}")

# Plot the regression line
plt.figure(figsize=(10, 6))

# Plot the regression line using the entire dataset for the line
plt.plot(df_budget_imdb['budget'], model.predict(X), color='red', linewidth=2, label='Regression Line')

# Set x-axis limits to focus on a more relevant range
plt.xlim(0, 200000000)  # Adjust this range as needed to fit better in the middle

plt.xlabel('Budget ($)')
plt.ylabel('IMDB Score')
plt.title('Regression Line for Budget vs IMDB Score')
plt.legend()
plt.grid(True)
plt.show()

# Calculate residuals for the second 100 points
residuals = df_budget_imdb['imdb_score'] - model.predict(sm.add_constant(df_budget_imdb['budget']))

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(df_budget_imdb['budget'], residuals, color='purple', alpha=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=2)  # Horizontal line at 0
plt.xlabel('Budget ($)')
plt.ylabel('Residuals')
plt.title('Residuals of IMDB Score Regression')
plt.grid(True)
plt.show()
