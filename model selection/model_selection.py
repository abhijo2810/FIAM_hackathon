import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2023-12-31')]
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    return df

def prepare_features(df, factors):
    available_factors = [f for f in factors if f in df.columns]
    print(f"Number of available factors: {len(available_factors)}")
    print(f"Factors not found in the dataset: {set(factors) - set(available_factors)}")
    
    if 'sentiment' in df.columns:
        available_factors.append('sentiment')
    else:
        print("Warning: 'sentiment' column not found in the dataset")
    
    X = df[available_factors]
    if 'stock_exret' in df.columns:
        y = df['stock_exret']
    else:
        print("Warning: 'stock_exret' column not found. Using a random target for demonstration.")
        y = np.random.randn(len(df))
    
    return X, y

def create_model_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def evaluate_model(model, X, y, train_index, test_index):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

def main():
    df = load_data('sentiment/data_with_sentiment.csv')
    with open('factor analysis/factor_char_list.csv', 'r') as f:
        factors = [line.strip() for line in f if line.strip()]
    
    X, y = prepare_features(df, factors)
    
    models = {
        'LASSO': create_model_pipeline(Lasso(alpha=0.1)),
        'Elastic Net': create_model_pipeline(ElasticNet(alpha=0.1, l1_ratio=0.5)),
        'Ridge': create_model_pipeline(Ridge(alpha=0.1))
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {model_name: {'mse': [], 'r2': []} for model_name in models.keys()}
    
    for train_index, test_index in tscv.split(X):
        for model_name, model in models.items():
            mse, r2 = evaluate_model(model, X, y, train_index, test_index)
            results[model_name]['mse'].append(mse)
            results[model_name]['r2'].append(r2)
    
    for model_name, result in results.items():
        print(f"{model_name} - Average MSE: {np.mean(result['mse']):.4f}, Average R2: {np.mean(result['r2']):.4f}")
    
    best_model_name = max(results, key=lambda x: np.mean(results[x]['r2']))
    best_model = models[best_model_name]
    best_model.fit(X, y)
    
    import joblib
    joblib.dump(best_model, 'model selection/best_model.joblib')
    print(f"Best model ({best_model_name}) saved to 'best_model.joblib'")

if __name__ == "__main__":
    main()