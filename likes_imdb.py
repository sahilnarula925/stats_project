# Import required packages
import pandas as pd
import statsmodels.api as sm
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'movie_metadata.csv'  # Ensure this file is in the same directory
if not os.path.exists(file_path):
    print(f"Error: Could not find data file at {file_path}")
    exit(1)

df = pd.read_csv(file_path, encoding='latin-1')

# Select relevant columns and drop missing values
df_likes_imdb = df[['movie_facebook_likes', 'imdb_score']].dropna()

# Convert columns to numeric if necessary
df_likes_imdb['movie_facebook_likes'] = pd.to_numeric(df_likes_imdb['movie_facebook_likes'], errors='coerce')
df_likes_imdb['imdb_score'] = pd.to_numeric(df_likes_imdb['imdb_score'], errors='coerce')

# Re-drop NAs after type conversion
df_likes_imdb = df_likes_imdb.dropna()

# Add a constant to the model (intercept)
X = sm.add_constant(df_likes_imdb['movie_facebook_likes'])
y = df_likes_imdb['imdb_score']

# Create the regression model
model = sm.OLS(y, X).fit()

# Display results clearly
print("\n" + "="*60)
print("Test 3: Linear Regression — Facebook Likes vs. IMDB Score")
print("="*60)

# Print the regression equation
print("\n📊 Model Summary:")
print(model.summary())

# Confidence Interval
ci = model.conf_int()
ci.columns = ['Lower CI', 'Upper CI']
print("\n🔍 95% Confidence Intervals:")
print(ci)

# Get relevant stats
slope = model.params['movie_facebook_likes']
p_value = model.pvalues['movie_facebook_likes']
r_squared = model.rsquared
slope_ci = ci.loc['movie_facebook_likes']

print("\n🧾 Interpretation:")
print(f"• Regression Coefficient (Slope): {slope:.4f}")
print(f"• R-squared: {r_squared:.4f}")
print(f"• 95% CI for Slope: ({slope_ci[0]:.4f}, {slope_ci[1]:.4f})")

# Hypothesis Testing Interpretation
alpha = 0.05
print("\n📌 Hypothesis Test:")
print("H₀: The number of Facebook likes and the IMDB score are not correlated.")
print("H₁: There is a statistically significant correlation between the number of Facebook likes and the IMDB score.")

if p_value < alpha:
    print(f"✅ p-value < {alpha}, reject the null hypothesis (H₀).")
    print("👉 Conclusion: There IS a statistically significant relationship between the number of Facebook likes and the IMDB score.")
else:
    print(f"❌ p-value ≥ {alpha}, fail to reject the null hypothesis (H₀).")
    print("👉 Conclusion: There is NO statistically significant relationship between the number of Facebook likes and the IMDB score.")

# Plot with regression line
plt.figure(figsize=(8, 5))
sns.regplot(x='movie_facebook_likes', y='imdb_score', data=df_likes_imdb, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Facebook Likes vs. IMDB Score')
plt.xlabel('Number of Facebook Likes')
plt.ylabel('IMDB Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate residuals
predicted_values = model.predict(X)
residuals = y - predicted_values

# Plot the residuals to check for normality
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--', linewidth=2)  # Vertical line at 0
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
