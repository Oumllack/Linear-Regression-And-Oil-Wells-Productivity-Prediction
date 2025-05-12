"""
Script to generate all visualizations for the README documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import shap
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data_and_models():
    """Load data and trained models."""
    data_path = Path('data/processed/features_engineered.csv')
    models_path = Path('models')
    
    df = pd.read_csv(data_path)
    xgb_model = joblib.load(models_path / 'xgboost.joblib')
    rf_model = joblib.load(models_path / 'random_forest.joblib')
    scaler = joblib.load(models_path / 'feature_scaler.joblib')
    
    return df, xgb_model, rf_model, scaler

def plot_feature_importance():
    """Generate feature importance plots for both models."""
    # Load importance data
    rf_importance = pd.read_csv('models/random_forest_importance.csv')
    xgb_importance = pd.read_csv('models/xgboost_importance.csv')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Random Forest plot
    sns.barplot(data=rf_importance, x='importance', y='feature', ax=ax1)
    ax1.set_title('Random Forest Feature Importance')
    ax1.set_xlabel('Importance (%)')
    
    # XGBoost plot
    sns.barplot(data=xgb_importance, x='importance', y='feature', ax=ax2)
    ax2.set_title('XGBoost Feature Importance')
    ax2.set_xlabel('Importance (%)')
    
    plt.tight_layout()
    plt.savefig('docs/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df):
    """Generate correlation matrix heatmap."""
    # Select numerical features
    numerical_cols = ['oil_rate', 'bottomhole_pressure', 'water_cut', 
                     'PI', 'cumulative_production', 'decline_rate']
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('docs/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_production_vs_prediction(df, model, scaler, model_name):
    """Generate production vs prediction plot for a sample well."""
    # Select a sample well
    sample_well = df['well_id'].unique()[0]
    well_data = df[df['well_id'] == sample_well].copy()
    
    # Prepare features
    feature_cols = ['oil_rate', 'bottomhole_pressure', 'water_cut',
                   'decline_rate', 'PI', 'cumulative_production',
                   'water_cut_trend', 'year', 'month']
    
    X = well_data[feature_cols]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(well_data['date'], well_data['oil_rate'], 
             label='Actual Production', color='blue')
    plt.plot(well_data['date'], predictions, 
             label='Predicted Production', color='red', linestyle='--')
    plt.title(f'Production vs Prediction - {model_name}\nWell {sample_well}')
    plt.xlabel('Date')
    plt.ylabel('Oil Rate (bbl/day)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'docs/figures/production_prediction_{model_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap_values(model, X_scaled, feature_names, model_name):
    """Generate SHAP summary plot."""
    # Calculate SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_scaled, 
                     feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}')
    plt.tight_layout()
    plt.savefig(f'docs/figures/shap_summary_{model_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curves(df, model, scaler, model_name):
    """Generate learning curves for the model."""
    from sklearn.model_selection import learning_curve
    
    # Prepare data
    feature_cols = ['oil_rate', 'bottomhole_pressure', 'water_cut',
                   'decline_rate', 'PI', 'cumulative_production',
                   'water_cut_trend', 'year', 'month']
    X = df[feature_cols]
    y = df['oil_rate']
    X_scaled = scaler.transform(X)
    
    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    # Calculate mean and std
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, 
                     test_mean + test_std, alpha=0.1)
    plt.title(f'Learning Curves - {model_name}')
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'docs/figures/learning_curves_{model_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations."""
    # Create figures directory
    Path('docs/figures').mkdir(parents=True, exist_ok=True)
    
    # Load data and models
    df, xgb_model, rf_model, scaler = load_data_and_models()
    
    # Generate all plots
    plot_feature_importance()
    plot_correlation_matrix(df)
    
    # Generate plots for both models
    for model, model_name in [(xgb_model, 'XGBoost'), (rf_model, 'Random Forest')]:
        plot_production_vs_prediction(df, model, scaler, model_name)
        plot_learning_curves(df, model, scaler, model_name)
        
        # Prepare data for SHAP
        feature_cols = ['oil_rate', 'bottomhole_pressure', 'water_cut',
                       'decline_rate', 'PI', 'cumulative_production',
                       'water_cut_trend', 'year', 'month']
        X = df[feature_cols].iloc[:100]  # Use subset for SHAP
        X_scaled = scaler.transform(X)
        plot_shap_values(model, X_scaled, feature_cols, model_name)

if __name__ == '__main__':
    main() 