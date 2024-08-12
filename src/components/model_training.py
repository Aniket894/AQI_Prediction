# src/components/model_training.py
from src.logger import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint
import pickle
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def train_models():
    logging.info("Started model training")

    try:
        # Load transformed datasets
        X_train = np.load('artifacts/X_train.npy')
        y_train = np.load('artifacts/y_train.npy')
        X_test = np.load('artifacts/X_test.npy')
        y_test = np.load('artifacts/y_test.npy')
        logging.info("Loaded transformed datasets")

        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Lasso': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': uniform(0.01, 100)
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': uniform(0.01, 100)
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': uniform(0.01, 100),
                    'l1_ratio': uniform(0.1, 0.9)
                }
            },
            'DecisionTreeRegressor': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {
                    'max_depth': [None, 3, 5, 7, 10],
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'max_depth': [None, 3, 5, 7, 10],
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'AdaBoostRegressor': {
                'model': AdaBoostRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'learning_rate': uniform(0.01, 1.0)
                }
            },
            'GradientBoostingRegressor': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 8)
                }
            },
            'XGBRegressor': {
                'model': XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': randint(50, 201),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 8)
                }
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': randint(3, 10),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
        }

        # Function to evaluate a regression model
        def evaluate_regression_model(model, params, X_train, y_train, X_test, y_test):
            logging.info(f"Starting RandomizedSearchCV for model: {model.__class__.__name__}")
            # Perform randomized search
            grid_search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
            grid_search.fit(X_train, y_train)
            
            logging.info(f"Completed RandomizedSearchCV for model: {model.__class__.__name__}")
            
            # Get the best model and its parameters
            best_model = grid_search.best_estimator_
            logging.info(f"Best model parameters: {grid_search.best_params_}")

            # Make predictions on the test set
            y_pred = best_model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            adjusted_r2 = 1 - (1-r2) * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

            logging.info(f"Model: {model.__class__.__name__}, MSE: {mse:.4f}, R²: {r2:.4f}, Adjusted R²: {adjusted_r2:.4f}")

            return best_model, mse, r2, adjusted_r2

        # Store best models and their performances
        best_models = {}
        for model_name, model_details in models.items():
            logging.info(f"Training {model_name}")
            best_model, mse, r2, adjusted_r2 = evaluate_regression_model(
                model_details['model'], model_details['params'], X_train, y_train, X_test, y_test
            )
            best_models[model_name] = {
                'model': best_model,
                'MSE': mse,
                'R²': r2,
                'Adjusted R²': adjusted_r2
            }
            # Save the best model
            model_path = f'artifacts/{model_name}_best_model.pkl'
            joblib.dump(best_model, model_path)
            logging.info(f"Saved best {model_name} model to {model_path}")

        # Save all models' performances for later comparison
        with open('artifacts/model_performances.pkl', 'wb') as f:
            pickle.dump(best_models, f)
        logging.info("Saved model performances to artifacts folder")

        # Visualize and save performance comparison
        plt.figure(figsize=(10, 6))
        for model_name, metrics in best_models.items():
            plt.bar(model_name, metrics['Adjusted R²'], label=f"{model_name}: {metrics['Adjusted R²']:.4f}")
        plt.ylabel('Adjusted R² Score')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('artifacts/model_performance_comparison.png')
        logging.info("Saved model performance comparison plot")

    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_models()
