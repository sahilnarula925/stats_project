# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# Load the dataset
try:
    # Try to find the data file in the current directory
    file_path = 'movie_metadata.csv'
    if not os.path.exists(file_path):
        print(f"Error: Could not find data file at {file_path}")
        print("Please make sure the data file is in the same directory as this script.")
        exit(1)
    
    df = pd.read_csv(file_path, encoding='latin-1')  # sometimes encoding issues occur with non-UTF8 files
    print("Successfully loaded the dataset!")
    
    # Preview the data structure
    print("\nData Shape:", df.shape)
    print("\nFirst few rows of data:")
    print(df.head())

    # Basic info and missing value analysis
    print("\nData Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Select the variables of interest:
    # For test1: budget and imdb_score
    # For test2: duration and num_critic_for_reviews (assumed to be the number of critic reviews)
    # For test3: movie_facebook_likes and imdb_score

    # Drop missing values for the selected columns (each analysis will drop missing rows as needed)
    df_budget_imdb = df[['budget', 'imdb_score']].dropna()
    df_duration_reviews = df[['duration', 'num_critic_for_reviews']].dropna()
    df_fb_imdb = df[['movie_facebook_likes', 'imdb_score']].dropna()

    # Convert columns to numeric if necessary (sometimes columns are object type)
    cols_to_numeric = ['budget', 'imdb_score', 'duration', 'num_critic_for_reviews', 'movie_facebook_likes']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Re-drop NAs after type conversion
    df_budget_imdb = df[['budget', 'imdb_score']].dropna()
    df_duration_reviews = df[['duration', 'num_critic_for_reviews']].dropna()
    df_fb_imdb = df[['movie_facebook_likes', 'imdb_score']].dropna()

    # Descriptive statistics
    print("\nSummary Statistics:")
    print("Budget & IMDB Score:")
    print(df_budget_imdb.describe())
    print("\nDuration & Critic Reviews:")
    print(df_duration_reviews.describe())
    print("\nFacebook Likes & IMDB Score:")
    print(df_fb_imdb.describe())

    # -----------------------------------------------------------------------------
    # FOCUS ON BUDGET VS IMDB SCORE LINEAR REGRESSION
    # -----------------------------------------------------------------------------
    print("\n*** DETAILED ANALYSIS: BUDGET VS IMDB SCORE ***")
    print("H₀: There is no association between the budget and the IMDB score.")
    print("H₁: There is a statistically significant relationship between the budget and the IMDB score.")
    
    # Add a constant to the model (intercept)
    X = sm.add_constant(df_budget_imdb['budget'])
    y = df_budget_imdb['imdb_score']
    
    # Create the regression model
    model = sm.OLS(y, X).fit()
    
    # Get the coefficients
    b0 = model.params[0]  # Intercept (B0)
    b1 = model.params[1]  # Slope (B1)
    
    # Print the model coefficients and equation
    print("\nLinear Regression Model: IMDB Score = B0 + B1 × Budget")
    print(f"Intercept (B0): {b0:.4f}")
    print(f"Slope (B1): {b1:.10f}")
    
    # Print the regression equation
    print(f"\nRegression Equation: IMDB Score = {b0:.4f} + {b1:.10f} × Budget")
    
    # Get R-squared and adjusted R-squared
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    print(f"\nR-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {adj_r_squared:.4f}")
    
    # Get the p-value for the budget coefficient
    p_value = model.pvalues[1]
    print(f"\nP-value for Budget coefficient: {p_value:.10f}")
    
    # Hypothesis test conclusion
    alpha = 0.05
    if p_value < alpha:
        print(f"\nSince p-value ({p_value:.10f}) is less than alpha ({alpha}), we REJECT the null hypothesis.")
        print("Conclusion: There IS a statistically significant relationship between budget and IMDB score.")
    else:
        print(f"\nSince p-value ({p_value:.10f}) is greater than or equal to alpha ({alpha}), we FAIL TO REJECT the null hypothesis.")
        print("Conclusion: There is NOT enough evidence to suggest a significant relationship between budget and IMDB score.")
    
    # Generate predictions using the model
    df_budget_imdb['predicted_score'] = model.predict(X)
    
    # Plot the first 50 points vs the regression line
    plt.figure(figsize=(10, 6))
    
    # Sort the dataframe by budget to make the regression line clearer
    df_plot = df_budget_imdb.sort_values('budget')
    
    # Get only first 50 points for scatter plot
    df_first_50 = df_budget_imdb.head(50)
    
    # Plot scatter points (first 50 points)
    #plt.scatter(df_first_50['budget'], df_first_50['imdb_score'], color='blue', alpha=0.7, label='First 50 Data Points')
    
    # Plot the regression line (using all data for the line)
    plt.plot(df_plot['budget'], df_plot['predicted_score'], color='red', linewidth=2, label=f'Regression Line: y = {b0:.4f} + {b1:.10f}x')
    
    # Add labels and title
    plt.xlabel('Budget ($)')
    plt.ylabel('IMDB Score')
    plt.title('Budget vs IMDB Score: Regression Line')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text with R-squared and p-value
    #plt.annotate(f"R² = {r_squared:.4f}\np-value = {p_value:.10f}", 
                 #xy=(0.05, 0.95), xycoords='axes fraction',
                 #bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print the summary of the model
    print("\nDetailed Regression Results:")
    print(model.summary())

    # -----------------------------------------------------------------------------
    # The rest of the original code continues below
    # -----------------------------------------------------------------------------

    # Visualization: Histograms
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(df_budget_imdb['budget'], bins=30, kde=True)
    plt.title('Budget Distribution')

    plt.subplot(1, 3, 2)
    sns.histplot(df_budget_imdb['imdb_score'], bins=30, kde=True)
    plt.title('IMDB Score Distribution')

    plt.subplot(1, 3, 3)
    sns.histplot(df_duration_reviews['duration'], bins=30, kde=True)
    plt.title('Duration Distribution')

    plt.tight_layout()
    plt.show()

    # Visualization: Scatter Plots with Regression Line
    # Test 1: Budget vs IMDB Score
    plt.figure(figsize=(6, 4))
    sns.regplot(x='budget', y='imdb_score', data=df_budget_imdb, scatter_kws={'alpha':0.5})
    plt.title('Budget vs IMDB Score')
    plt.xlabel('Budget')
    plt.ylabel('IMDB Score')
    plt.show()

    # Test 2: Duration vs Number of Critic Reviews
    plt.figure(figsize=(6, 4))
    sns.regplot(x='duration', y='num_critic_for_reviews', data=df_duration_reviews, scatter_kws={'alpha':0.5})
    plt.title('Duration vs Critic Reviews')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Number of Critic Reviews')
    plt.show()

    # Test 3: Facebook Likes vs IMDB Score
    plt.figure(figsize=(6, 4))
    sns.regplot(x='movie_facebook_likes', y='imdb_score', data=df_fb_imdb, scatter_kws={'alpha':0.5})
    plt.title('Facebook Likes vs IMDB Score')
    plt.xlabel('Facebook Likes')
    plt.ylabel('IMDB Score')
    plt.show()

    # Box Plots to visualize potential outliers (example for budget)
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df_budget_imdb['budget'])
    plt.title('Box Plot of Budget')
    plt.show()

    # Note: Identify and treat outliers as necessary (e.g., by removing values above a threshold)
    # Here, for demonstration, you could filter out extremely high budgets if they are anomalies.
    budget_threshold = df_budget_imdb['budget'].quantile(0.99)
    df_budget_imdb_filtered = df_budget_imdb[df_budget_imdb['budget'] <= budget_threshold]

    # Re-run scatter plot on filtered data if necessary
    plt.figure(figsize=(6, 4))
    sns.regplot(x='budget', y='imdb_score', data=df_budget_imdb_filtered, scatter_kws={'alpha':0.5})
    plt.title('Budget vs IMDB Score (Filtered)')
    plt.xlabel('Budget')
    plt.ylabel('IMDB Score')
    plt.show()

    # -----------------------------------------------------------------------------
    # Statistical Tests with Linear Regression
    # -----------------------------------------------------------------------------

    # Test 1: Impact of Budget on IMDB Score
    # H0: There is no significant relationship between budget and IMDB score.
    # H1: There is a significant relationship between budget and IMDB score.

    # Linear Regression model: imdb_score ~ budget
    model_budget = smf.ols('imdb_score ~ budget', data=df_budget_imdb).fit()
    print("\n--- Regression Results: Budget vs IMDB Score ---")
    print(model_budget.summary())

    # Interpretation: Look at the p-value for the budget coefficient.
    # If p-value < 0.05, reject H0 at the 5% significance level.

    # Test 2: Relationship between Duration and Number of Critic Reviews
    # H0: There is no significant relationship between movie duration and the number of critic reviews.
    # H1: There is a significant relationship between movie duration and the number of critic reviews.

    model_duration = smf.ols('num_critic_for_reviews ~ duration', data=df_duration_reviews).fit()
    print("\n--- Regression Results: Duration vs Critic Reviews ---")
    print(model_duration.summary())

    # Interpretation: Assess the p-value for the duration variable.

    # Test 3: Correlation between Facebook Likes and IMDB Score
    # H0: There is no significant correlation between Facebook likes and IMDB score.
    # H1: There is a significant correlation between Facebook likes and IMDB score.

    model_fb = smf.ols('imdb_score ~ movie_facebook_likes', data=df_fb_imdb).fit()
    print("\n--- Regression Results: Facebook Likes vs IMDB Score ---")
    print(model_fb.summary())

    # -----------------------------------------------------------------------------
    # Conclusions from Statistical Tests
    # -----------------------------------------------------------------------------
    # Based on the regression outputs:
    # - For Test 1: If the budget coefficient's p-value < 0.05, budget is a significant predictor of imdb_score.
    # - For Test 2: Similarly, duration is significant if p-value < 0.05.
    # - For Test 3: A statistically significant relationship exists if the p-value for movie_facebook_likes is less than 0.05.

    # Confidence intervals for each coefficient can be interpreted from the model summary.
    # For example:
    print("\nConfidence intervals for Test 1 (Budget vs IMDB Score):")
    print(model_budget.conf_int())

    # -----------------------------------------------------------------------------
    # End of Code
    # -----------------------------------------------------------------------------
except Exception as e:
    print(f"Error: {e}")