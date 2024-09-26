import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DropAllNaNFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_to_drop_ = X.columns[X.isnull().all()].tolist()
        if self.columns_to_drop_:
            logging.warning(f"Features with all NaN values that will be dropped: {self.columns_to_drop_}")
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

def load_factors(file_path):
    """Load factor list from a file"""
    with open(file_path, 'r') as f:
        factors = [line.strip() for line in f if line.strip()]
    return factors

def load_and_prepare_data(file_path, factors):
    """Load and prepare the data for modeling"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date')
    
    logging.info(f"Columns in the dataset: {df.columns.tolist()}")
    logging.info(f"Number of rows: {len(df)}")
    
    # Check which factors are actually in the dataset
    available_factors = [f for f in factors if f in df.columns]
    missing_factors = set(factors) - set(available_factors)
    
    logging.info(f"Number of factors in the list: {len(factors)}")
    logging.info(f"Number of available factors: {len(available_factors)}")
    logging.info(f"Missing factors: {missing_factors}")
    
    X = df[available_factors]
    y = df['stock_exret']
    
    # Print information about missing values
    missing_percentages = (X.isnull().sum() / len(X)) * 100
    logging.info("\nMissing value percentages in features:")
    logging.info(missing_percentages.sort_values(ascending=False).head())
    logging.info("\nMissing values in target:")
    logging.info(y.isnull().sum())
    
    return X, y, df['date'], available_factors

def create_model_pipeline(model):
    """Create a pipeline with custom transformer, imputation, scaling, and the model"""
    return Pipeline([
        ('drop_all_nan', DropAllNaNFeatures()),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def train_and_evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """Train and evaluate a model pipeline"""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, pipeline

def rolling_window_cv(X, y, dates, window_size, step_size, models):
    """Perform rolling window cross-validation"""
    results = {model_name: {'mse': [], 'r2': [], 'models': []} for model_name in models.keys()}
    evaluation_dates = []
    
    for start_idx in range(0, len(X) - window_size - step_size, step_size):
        train_end = start_idx + window_size
        test_end = train_end + step_size
        
        X_train, y_train = X.iloc[start_idx:train_end], y.iloc[start_idx:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
        
        evaluation_dates.append(dates.iloc[test_end - 1])
        
        for model_name, model in models.items():
            mse, r2, trained_pipeline = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            results[model_name]['mse'].append(mse)
            results[model_name]['r2'].append(r2)
            results[model_name]['models'].append(trained_pipeline)
    
    return results, evaluation_dates

def plot_performance(results, dates):
    """Plot model performance over time"""
    plt.figure(figsize=(12, 6))
    for model_name, result in results.items():
        plt.plot(dates, result['r2'], label=model_name)
    
    plt.title('Model Performance (R2) Over Time')
    plt.xlabel('Date')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()

def main():
    # Load factor list
    factors = load_factors('factor_char_list.csv')
    
    # Load and prepare data
    X, y, dates, available_factors = load_and_prepare_data('hackathon_sample_v2.csv', factors)
    
    # Define models with pipelines
    models = {
        'LASSO': create_model_pipeline(Lasso(alpha=0.1, random_state=42)),
        'Elastic Net': create_model_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
        'Ridge': create_model_pipeline(Ridge(alpha=0.1, random_state=42))
    }
    
    # Perform rolling window cross-validation
    window_size = 60  # 5 years of monthly data
    step_size = 12    # 1 year step
    results, evaluation_dates = rolling_window_cv(X, y, dates, window_size, step_size, models)
    
    # Plot performance
    plot_performance(results, evaluation_dates)
    
    # Print average performance
    for model_name, result in results.items():
        logging.info(f"{model_name} - Average MSE: {np.mean(result['mse']):.4f}, Average R2: {np.mean(result['r2']):.4f}")
    
    # Save the best model (you can modify this criteria based on your preference)
    best_model_name = max(results, key=lambda x: np.mean(results[x]['r2']))
    best_pipeline = results[best_model_name]['models'][-1]
    
    import joblib
    joblib.dump(best_pipeline, 'best_model_pipeline.joblib')
    logging.info(f"Best model pipeline ({best_model_name}) saved to 'best_model_pipeline.joblib'")
    
    # Save feature importance for the best model
    best_model = best_pipeline.named_steps['model']
    if hasattr(best_model, 'coef_'):
        # Get the feature names after dropping all-NaN features
        final_features = best_pipeline.named_steps['drop_all_nan'].transform(X).columns.tolist()
        feature_importance = pd.DataFrame({
            'feature': final_features,
            'importance': np.abs(best_model.coef_)
        }).sort_values('importance', ascending=False)
        feature_importance.to_csv('feature_importance.csv', index=False)
        logging.info("Feature importance saved to 'feature_importance.csv'")

if __name__ == "__main__":
    main()