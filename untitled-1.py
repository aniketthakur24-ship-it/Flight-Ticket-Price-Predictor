"""
Airline prediction model using machine learning techniques. This script loads flight ticket price data, preprocesses it, trains multiple regression models, evaluates their performance, and visualizes the results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("AIRLINE TICKET PRICE PREDICTION MODEL")
print("=" * 60)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================
print("\n[1] Loading Data...")
df = pd.read_csv(r"C:\Users\at396\OneDrive\Desktop\Python\Projects\flight_ticket_price_data.csv.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2] Data Preprocessing...")

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Airline', 'Origin', 'Destination', 'Class']

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Separate features and target
X = df_processed.drop(['Ticket_ID', 'Price_USD'], axis=1)
y = df_processed['Price_USD']

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Target variable: Price_USD")
print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. MODEL TRAINING AND EVALUATION
# ============================================================================
print("\n[3] Training Models...")
print("-" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    if name == 'Linear Regression':
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                    scoring='r2')
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                    scoring='r2')
    
    results[name] = {
        'model': model,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 4. MODEL COMPARISON AND RESULTS
# ============================================================================
print("\n[4] Model Comparison Results")
print("-" * 60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[m]['mae'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'R² Score': [results[m]['r2'] for m in results.keys()],
    'CV R² (Mean)': [results[m]['cv_mean'] for m in results.keys()]
})

print("\n" + comparison_df.to_string(index=False))

# Best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"R² Score: {results[best_model_name]['r2']:.4f}")
print(f"RMSE: ${results[best_model_name]['rmse']:.2f}")
print(f"{'='*60}")

# ============================================================================
# 5. FEATURE IMPORTANCE (for tree-based models)
# ============================================================================
print("\n[5] Feature Importance Analysis")
print("-" * 60)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 6. SAMPLE PREDICTIONS
# ============================================================================
print("\n[6] Sample Predictions on Test Set")
print("-" * 60)

# Show first 10 test predictions
sample_predictions = pd.DataFrame({
    'Actual Price': y_test.values[:10],
    'Predicted Price': results[best_model_name]['predictions'][:10],
    'Difference': (y_test.values[:10] - results[best_model_name]['predictions'][:10]),
    'Error %': np.abs((y_test.values[:10] - results[best_model_name]['predictions'][:10]) / 
                      y_test.values[:10] * 100)
})

print("\n" + sample_predictions.to_string(index=False))

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[7] Generating Visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_test, results[best_model_name]['predictions'], alpha=0.6, edgecolors='k')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Price ($)', fontsize=11)
ax1.set_ylabel('Predicted Price ($)', fontsize=11)
ax1.set_title(f'{best_model_name}: Actual vs Predicted Prices', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Residuals
ax2 = axes[0, 1]
residuals = y_test - results[best_model_name]['predictions']
ax2.scatter(results[best_model_name]['predictions'], residuals, alpha=0.6, edgecolors='k')
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Price ($)', fontsize=11)
ax2.set_ylabel('Residuals ($)', fontsize=11)
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Model Comparison
ax3 = axes[1, 0]
model_names = list(results.keys())
r2_scores = [results[m]['r2'] for m in model_names]
colors = ['#2ecc71' if m == best_model_name else '#3498db' for m in model_names]
bars = ax3.bar(model_names, r2_scores, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('R² Score', fontsize=11)
ax3.set_title('Model Performance Comparison (R² Score)', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1])
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Feature Importance (if available)
ax4 = axes[1, 1]
if hasattr(best_model, 'feature_importances_'):
    top_features = feature_importance.head(10)
    ax4.barh(range(len(top_features)), top_features['Importance'].values, 
             color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['Feature'].values)
    ax4.set_xlabel('Importance', fontsize=11)
    ax4.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
else:
    ax4.text(0.5, 0.5, 'Feature Importance\nNot Available', 
            ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.axis('off')

plt.tight_layout()
plt.savefig('airline_price_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'airline_price_prediction_analysis.png'")

plt.show()

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("\n[8] Summary Statistics")
print("-" * 60)
print(f"\nPrice Range: ${y.min():.2f} - ${y.max():.2f}")
print(f"Mean Price: ${y.mean():.2f}")
print(f"Std Dev: ${y.std():.2f}")
print(f"\nTest Set Mean Absolute Error: ${results[best_model_name]['mae']:.2f}")
print(f"Average Prediction Error: {np.mean(np.abs(residuals/y_test)*100):.2f}%")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
import joblib

joblib.dump(best_model, "flight_price_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
print("Model saved successfully!")






import os

print("Current Folder:", os.getcwd())
print("Files in Folder:")
print(os.listdir())

