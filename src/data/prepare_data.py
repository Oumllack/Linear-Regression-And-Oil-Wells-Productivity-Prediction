"""
Module de préparation des données pour le projet de prédiction de productivité des puits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparator:
    """Classe pour la préparation et le nettoyage des données de production."""
    
    def __init__(self, data_path: str):
        """
        Initialise le préparateur de données.
        
        Args:
            data_path (str): Chemin vers le dossier contenant les données
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.
        
        Args:
            filename (str): Nom du fichier à charger
            
        Returns:
            pd.DataFrame: Données chargées
        """
        try:
            file_path = self.data_path / 'raw' / filename
            logger.info(f"Chargement des données depuis {file_path}")
            self.raw_data = pd.read_csv(file_path)
            return self.raw_data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise
    
    def remove_outliers(self, df: pd.DataFrame, columns: list, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Supprime les valeurs aberrantes des colonnes spécifiées.
        
        Args:
            df (pd.DataFrame): DataFrame à nettoyer
            columns (list): Liste des colonnes à traiter
            method (str): Méthode de détection ('iqr' ou 'zscore')
            threshold (float): Seuil pour la détection
            
        Returns:
            pd.DataFrame: DataFrame nettoyé
        """
        df_clean = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                mask = z_scores < threshold
            else:
                raise ValueError("Méthode non supportée. Utilisez 'iqr' ou 'zscore'")
            
            df_clean = df_clean[mask]
            logger.info(f"Suppression de {len(df) - len(df_clean)} valeurs aberrantes dans {col}")
        
        return df_clean
    
    def normalize_features(self, df: pd.DataFrame, columns: list, 
                          method: str = 'standard') -> Tuple[pd.DataFrame, dict]:
        """
        Normalise les features spécifiées.
        
        Args:
            df (pd.DataFrame): DataFrame à normaliser
            columns (list): Liste des colonnes à normaliser
            method (str): Méthode de normalisation ('standard' ou 'minmax')
            
        Returns:
            Tuple[pd.DataFrame, dict]: DataFrame normalisé et paramètres de normalisation
        """
        df_norm = df.copy()
        scaler_params = {}
        
        for col in columns:
            if method == 'standard':
                mean = df[col].mean()
                std = df[col].std()
                df_norm[col] = (df[col] - mean) / std
                scaler_params[col] = {'mean': mean, 'std': std}
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                scaler_params[col] = {'min': min_val, 'max': max_val}
            else:
                raise ValueError("Méthode non supportée. Utilisez 'standard' ou 'minmax'")
        
        return df_norm, scaler_params
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'interpolate',
                            columns: Optional[list] = None) -> pd.DataFrame:
        """
        Gère les valeurs manquantes dans le DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame à traiter
            strategy (str): Stratégie de traitement ('interpolate', 'mean', 'median', 'drop')
            columns (list, optional): Liste des colonnes à traiter
            
        Returns:
            pd.DataFrame: DataFrame traité
        """
        df_clean = df.copy()
        columns = columns or df.columns
        
        for col in columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"Traitement de {missing_count} valeurs manquantes dans {col}")
                
                if strategy == 'interpolate':
                    df_clean[col] = df[col].interpolate(method='linear')
                elif strategy == 'mean':
                    df_clean[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df_clean[col] = df[col].fillna(df[col].median())
                elif strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                else:
                    raise ValueError("Stratégie non supportée")
        
        return df_clean
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Sauvegarde les données traitées.
        
        Args:
            df (pd.DataFrame): DataFrame à sauvegarder
            filename (str): Nom du fichier de sortie
        """
        output_path = self.data_path / 'processed' / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Données sauvegardées dans {output_path}")

def main():
    """Fonction principale pour la préparation des données."""
    # Exemple d'utilisation
    preparator = DataPreparator('data')
    
    # À adapter selon les noms de fichiers réels
    try:
        # Chargement des données
        df = preparator.load_data('production_data.csv')
        
        # Nettoyage des données
        df = preparator.handle_missing_values(df, strategy='interpolate')
        
        # Suppression des valeurs aberrantes
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = preparator.remove_outliers(df, columns=numeric_columns)
        
        # Normalisation des features
        df, scaler_params = preparator.normalize_features(
            df, 
            columns=numeric_columns,
            method='standard'
        )
        
        # Sauvegarde des données traitées
        preparator.save_processed_data(df, 'processed_production_data.csv')
        
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données: {str(e)}")
        raise

if __name__ == "__main__":
    main() 