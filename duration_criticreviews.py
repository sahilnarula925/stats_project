import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os

# Load the dataset
file_path = 'movie_metadata.csv'
if not os.path.exists(file_path):
    print(f"Error: Could not find data file at {file_path}")
    exit(1)

df = pd.read_csv(file_path, encoding='latin-1')

# Keep only necessary columns and drop missing values
df = df[['duration', 'num_critic_for_reviews']].dropna()

# Ensure columns are numeric
df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
df['num_critic_for_reviews'] = pd.to_numeric(df['num_critic_for_reviews'], errors='coerce')
df = df.dropna()

# Simple Linear Regression
model = smf.ols('num_critic_for_reviews ~ duration', data=df).fit()

# Display results clearly
print("\n" + "="*60)
print("Test 2: Linear Regression — Duration vs. Number of Critic Reviews")
print("="*60)

print("\n📊 Model Summary:")
print(model.summary())

# Confidence Interval
ci = model.conf_int()
ci.columns = ['Lower CI', 'Upper CI']
print("\n🔍 95% Confidence Intervals:")
print(ci)

# Get relevant stats
slope = model.params['duration']
p_value = model.pvalues['duration']
r_squared = model.rsquared
slope_ci = ci.loc['duration']

print("\n🧾 Interpretation:")
print(f"• Regression Coefficient (Slope): {slope:.4f}")
print(p_value)
print(f"• R-squared: {r_squared:.4f}")
print(f"• 95% CI for Slope: ({slope_ci[0]:.4f}, {slope_ci[1]:.4f})")

# Hypothesis Testing Interpretation
alpha = 0.05
print("\n📌 Hypothesis Test:")
print("H₀: Duration and number of critic reviews are independent.")
print("H₁: Duration significantly predicts the number of critic reviews.")

if p_value < alpha:
    print(f"✅ p-value < {alpha}, reject the null hypothesis (H₀).")
    print("👉 Conclusion: There IS a statistically significant relationship between movie duration and the number of critic reviews.")
else:
    print(f"❌ p-value ≥ {alpha}, fail to reject the null hypothesis (H₀).")
    print("👉 Conclusion: There is NO statistically significant relationship between movie duration and the number of critic reviews.")

# Plot with regression line
plt.figure(figsize=(8, 5))
sns.regplot(x='duration', y='num_critic_for_reviews', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Duration vs. Number of Critic Reviews')
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Critic Reviews')
plt.grid(True)
plt.tight_layout()
plt.show()

# Add a constant to the model (intercept)
X = sm.add_constant(df['duration'])
y = df['num_critic_for_reviews']

# Create the regression model
model = sm.OLS(y, X).fit()

# Get the predicted values
predicted_values = model.predict(X)

# Plot the regression line
plt.figure(figsize=(10, 6))

# Plot only the points that fit the regression line
plt.plot(df['duration'], predicted_values, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Critic Reviews')
plt.title('Regression Line for Duration vs. Number of Critic Reviews')
plt.legend()
plt.grid(True)
plt.show()

# Calculate residuals
residuals = y - predicted_values

# Plot the residuals
plt.figure(figsize=(10, 6))
plt.scatter(predicted_values, residuals, color='purple', alpha=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=2)  # Horizontal line at 0
plt.xlabel('Predicted Number of Critic Reviews')
plt.ylabel('Residuals')
plt.title('Residuals of the Regression Model')
plt.grid(True)
plt.tight_layout()
plt.show()
