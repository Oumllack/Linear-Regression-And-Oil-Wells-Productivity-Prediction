"""
Module d'entraînement des modèles pour la prédiction de productivité des puits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import shap

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Classe pour l'entraînement et l'évaluation des modèles."""
    
    def __init__(self, data_path: str, model_path: str):
        """
        Initialise le trainer de modèles.
        
        Args:
            data_path (str): Chemin vers le dossier des données
            model_path (str): Chemin vers le dossier de sauvegarde des modèles
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Créer le dossier models s'il n'existe pas
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str,
                    feature_cols: List[str],
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            target_col (str): Nom de la colonne cible
            feature_cols (List[str]): Liste des colonnes features
            test_size (float): Proportion des données de test
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            Tuple[np.ndarray, ...]: X_train, X_test, y_train, y_test
        """
        # Séparer features et target
        X = df[feature_cols]
        y = df[target_col]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalisation des features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['feature_scaler'] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     param_grid: Optional[Dict] = None) -> xgb.XGBRegressor:
        """
        Entraîne un modèle XGBoost avec optimisation des hyperparamètres.
        
        Args:
            X_train (np.ndarray): Features d'entraînement
            y_train (np.ndarray): Target d'entraînement
            param_grid (Dict, optional): Grille de paramètres pour GridSearchCV
            
        Returns:
            xgb.XGBRegressor: Modèle entraîné
        """
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        # Initialiser le modèle
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Début de l'entraînement XGBoost avec GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        # Récupérer le meilleur modèle
        best_model = grid_search.best_estimator_
        self.models['xgboost'] = best_model
        
        logger.info(f"Meilleurs paramètres XGBoost: {grid_search.best_params_}")
        
        return best_model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           param_grid: Optional[Dict] = None) -> RandomForestRegressor:
        """
        Entraîne un modèle Random Forest avec optimisation des hyperparamètres.
        
        Args:
            X_train (np.ndarray): Features d'entraînement
            y_train (np.ndarray): Target d'entraînement
            param_grid (Dict, optional): Grille de paramètres pour GridSearchCV
            
        Returns:
            RandomForestRegressor: Modèle entraîné
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        # Initialiser le modèle
        rf_model = RandomForestRegressor(random_state=42)
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Début de l'entraînement Random Forest avec GridSearchCV...")
        grid_search.fit(X_train, y_train)
        
        # Récupérer le meilleur modèle
        best_model = grid_search.best_estimator_
        self.models['random_forest'] = best_model
        
        logger.info(f"Meilleurs paramètres Random Forest: {grid_search.best_params_}")
        
        return best_model
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Évalue les performances d'un modèle.
        
        Args:
            model_name (str): Nom du modèle à évaluer
            X_test (np.ndarray): Features de test
            y_test (np.ndarray): Target de test
            
        Returns:
            Dict[str, float]: Dictionnaire des métriques d'évaluation
        """
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Métriques pour {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics
    
    def analyze_feature_importance(self, model_name: str, 
                                 feature_names: List[str]) -> pd.DataFrame:
        """
        Analyse l'importance des features pour un modèle.
        
        Args:
            model_name (str): Nom du modèle à analyser
            feature_names (List[str]): Liste des noms des features
            
        Returns:
            pd.DataFrame: DataFrame avec l'importance des features
        """
        model = self.models[model_name]
        
        if model_name == 'xgboost':
            importance = model.feature_importances_
        elif model_name == 'random_forest':
            importance = model.feature_importances_
        else:
            raise ValueError(f"Type de modèle non supporté: {model_name}")
        
        # Créer DataFrame d'importance des features
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        self.feature_importance[model_name] = importance_df
        
        return importance_df
    
    def explain_predictions(self, model_name: str, X_test: np.ndarray,
                          feature_names: List[str]) -> shap.Explanation:
        """
        Génère des explications SHAP pour les prédictions.
        
        Args:
            model_name (str): Nom du modèle à expliquer
            X_test (np.ndarray): Features de test
            feature_names (List[str]): Liste des noms des features
            
        Returns:
            shap.Explanation: Objet d'explication SHAP
        """
        model = self.models[model_name]
        
        # Créer l'explainer SHAP
        explainer = shap.Explainer(model)
        
        # Calculer les valeurs SHAP
        shap_values = explainer(X_test)
        
        return shap_values
    
    def save_model(self, model_name: str) -> None:
        """
        Sauvegarde un modèle entraîné.
        
        Args:
            model_name (str): Nom du modèle à sauvegarder
        """
        model = self.models[model_name]
        model_path = self.model_path / f"{model_name}.joblib"
        
        # Sauvegarder le modèle
        joblib.dump(model, model_path)
        
        # Sauvegarder les métriques et l'importance des features
        metrics_path = self.model_path / f"{model_name}_metrics.json"
        importance_path = self.model_path / f"{model_name}_importance.csv"
        
        if model_name in self.feature_importance:
            self.feature_importance[model_name].to_csv(importance_path, index=False)
        
        logger.info(f"Modèle {model_name} sauvegardé dans {model_path}")
    
    def save_scaler(self, scaler_name: str) -> None:
        """
        Sauvegarde un scaler.
        
        Args:
            scaler_name (str): Nom du scaler à sauvegarder
        """
        scaler = self.scalers[scaler_name]
        scaler_path = self.model_path / f"{scaler_name}.joblib"
        
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler {scaler_name} sauvegardé dans {scaler_path}")

def main():
    """Fonction principale pour l'entraînement des modèles."""
    # Exemple d'utilisation
    trainer = ModelTrainer('data', 'models')
    
    try:
        # Charger les données avec les features
        df = pd.read_csv(Path('data/processed/features_engineered.csv'))
        
        # Définir les colonnes features et target
        feature_cols = [
            'oil_rate', 'bottomhole_pressure', 'water_cut',
            'decline_rate', 'PI', 'cumulative_production',
            'water_cut_trend', 'year', 'month'
        ]
        target_col = 'oil_rate_next_month'  # À adapter selon les données
        
        # Préparer les données
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            df, target_col, feature_cols
        )
        
        # Entraîner XGBoost
        xgb_model = trainer.train_xgboost(X_train, y_train)
        xgb_metrics = trainer.evaluate_model('xgboost', X_test, y_test)
        xgb_importance = trainer.analyze_feature_importance('xgboost', feature_cols)
        
        # Entraîner Random Forest
        rf_model = trainer.train_random_forest(X_train, y_train)
        rf_metrics = trainer.evaluate_model('random_forest', X_test, y_test)
        rf_importance = trainer.analyze_feature_importance('random_forest', feature_cols)
        
        # Générer les explications SHAP
        xgb_shap = trainer.explain_predictions('xgboost', X_test, feature_cols)
        rf_shap = trainer.explain_predictions('random_forest', X_test, feature_cols)
        
        # Sauvegarder les modèles et les scalers
        trainer.save_model('xgboost')
        trainer.save_model('random_forest')
        trainer.save_scaler('feature_scaler')
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement des modèles: {str(e)}")
        raise

if __name__ == "__main__":
    main() 