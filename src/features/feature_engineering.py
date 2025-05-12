"""
Module de feature engineering pour le projet de prédiction de productivité des puits.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Classe pour la création et la transformation des features."""
    
    def __init__(self, data_path: str):
        """
        Initialise l'ingénieur de features.
        
        Args:
            data_path (str): Chemin vers le dossier contenant les données
        """
        self.data_path = Path(data_path)
        self.feature_params = {}
    
    def calculate_decline_rate(self, df: pd.DataFrame, 
                             production_col: str,
                             time_col: str,
                             well_id_col: str,
                             window: int = 30) -> pd.DataFrame:
        """
        Calcule le taux de déclin de production pour chaque puits.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            production_col (str): Nom de la colonne de production
            time_col (str): Nom de la colonne de temps
            well_id_col (str): Nom de la colonne d'identifiant de puits
            window (int): Fenêtre de temps pour le calcul (en jours)
            
        Returns:
            pd.DataFrame: DataFrame avec la colonne de taux de déclin ajoutée
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Trier par puits et par date
        df = df.sort_values([well_id_col, time_col])
        
        # Calculer le taux de déclin
        decline_rates = []
        for well in df[well_id_col].unique():
            well_data = df[df[well_id_col] == well].copy()
            
            # Calculer la moyenne mobile
            well_data['rolling_mean'] = well_data[production_col].rolling(window=window).mean()
            
            # Calculer le taux de déclin
            well_data['decline_rate'] = -well_data['rolling_mean'].pct_change(periods=window)
            
            decline_rates.append(well_data)
        
        df_with_decline = pd.concat(decline_rates)
        return df_with_decline
    
    def calculate_productivity_index(self, df: pd.DataFrame,
                                   production_col: str,
                                   pressure_col: str,
                                   well_id_col: str) -> pd.DataFrame:
        """
        Calcule l'indice de productivité (PI) pour chaque puits.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            production_col (str): Nom de la colonne de production
            pressure_col (str): Nom de la colonne de pression
            well_id_col (str): Nom de la colonne d'identifiant de puits
            
        Returns:
            pd.DataFrame: DataFrame avec la colonne PI ajoutée
        """
        df = df.copy()
        
        # Calculer PI = Q / (P_initial - P_actuelle)
        df['PI'] = df[production_col] / (df.groupby(well_id_col)[pressure_col].transform('first') - df[pressure_col])
        
        # Gérer les divisions par zéro
        df['PI'] = df['PI'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def calculate_cumulative_production(self, df: pd.DataFrame,
                                      production_col: str,
                                      time_col: str,
                                      well_id_col: str) -> pd.DataFrame:
        """
        Calcule la production cumulative pour chaque puits.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            production_col (str): Nom de la colonne de production
            time_col (str): Nom de la colonne de temps
            well_id_col (str): Nom de la colonne d'identifiant de puits
            
        Returns:
            pd.DataFrame: DataFrame avec la colonne de production cumulative ajoutée
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Trier par puits et par date
        df = df.sort_values([well_id_col, time_col])
        
        # Calculer la production cumulative
        df['cumulative_production'] = df.groupby(well_id_col)[production_col].cumsum()
        
        return df
    
    def calculate_water_cut_trend(self, df: pd.DataFrame,
                                water_cut_col: str,
                                time_col: str,
                                well_id_col: str,
                                window: int = 30) -> pd.DataFrame:
        """
        Calcule la tendance du water cut pour chaque puits.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            water_cut_col (str): Nom de la colonne de water cut
            time_col (str): Nom de la colonne de temps
            well_id_col (str): Nom de la colonne d'identifiant de puits
            window (int): Fenêtre de temps pour le calcul (en jours)
            
        Returns:
            pd.DataFrame: DataFrame avec la colonne de tendance du water cut ajoutée
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Trier par puits et par date
        df = df.sort_values([well_id_col, time_col])
        
        # Calculer la tendance du water cut
        water_cut_trends = []
        for well in df[well_id_col].unique():
            well_data = df[df[well_id_col] == well].copy()
            
            # Calculer la moyenne mobile
            well_data['water_cut_trend'] = well_data[water_cut_col].rolling(window=window).mean()
            
            water_cut_trends.append(well_data)
        
        df_with_trend = pd.concat(water_cut_trends)
        return df_with_trend
    
    def create_time_features(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """
        Crée des features temporelles à partir de la colonne de temps.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            time_col (str): Nom de la colonne de temps
            
        Returns:
            pd.DataFrame: DataFrame avec les features temporelles ajoutées
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Extraire les composantes temporelles
        df['year'] = df[time_col].dt.year
        df['month'] = df[time_col].dt.month
        df['day'] = df[time_col].dt.day
        df['dayofweek'] = df[time_col].dt.dayofweek
        df['quarter'] = df[time_col].dt.quarter
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                  feature_pairs: List[tuple]) -> pd.DataFrame:
        """
        Crée des features d'interaction entre des paires de variables.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données
            feature_pairs (List[tuple]): Liste de paires de colonnes pour créer des interactions
            
        Returns:
            pd.DataFrame: DataFrame avec les features d'interaction ajoutées
        """
        df = df.copy()
        
        for col1, col2 in feature_pairs:
            # Multiplication
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # Ratio (avec gestion des divisions par zéro)
            df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
            
            # Différence
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, well_id_col: str, date_col: str, target_col: str) -> pd.DataFrame:
        """Crée la variable cible en décalant la production d'un mois."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Trier les données par puits et par date
        df = df.sort_values([well_id_col, date_col])
        
        # Créer la variable cible en décalant la production d'un mois
        df[f'{target_col}_next_month'] = df.groupby(well_id_col)[target_col].shift(-1)
        
        # Supprimer la dernière ligne de chaque puits car nous n'avons pas la valeur suivante
        df = df.groupby(well_id_col).apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        
        return df
    
    def save_features(self, df: pd.DataFrame, filename: str) -> None:
        """
        Sauvegarde les données avec les nouvelles features.
        
        Args:
            df (pd.DataFrame): DataFrame à sauvegarder
            filename (str): Nom du fichier de sortie
        """
        output_path = self.data_path / 'processed' / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Features sauvegardées dans {output_path}")

def main():
    """Fonction principale pour la création des features."""
    # Exemple d'utilisation
    engineer = FeatureEngineer('data')
    
    try:
        # Charger les données traitées
        df = pd.read_csv(Path('data/processed/processed_production_data.csv'))
        
        # Créer les features
        df = engineer.calculate_decline_rate(
            df,
            production_col='oil_rate',
            time_col='date',
            well_id_col='well_id'
        )
        
        df = engineer.calculate_productivity_index(
            df,
            production_col='oil_rate',
            pressure_col='bottomhole_pressure',
            well_id_col='well_id'
        )
        
        df = engineer.calculate_cumulative_production(
            df,
            production_col='oil_rate',
            time_col='date',
            well_id_col='well_id'
        )
        
        df = engineer.calculate_water_cut_trend(
            df,
            water_cut_col='water_cut',
            time_col='date',
            well_id_col='well_id'
        )
        
        df = engineer.create_time_features(df, time_col='date')
        
        # Créer des features d'interaction
        interaction_pairs = [
            ('oil_rate', 'bottomhole_pressure'),
            ('water_cut', 'oil_rate'),
            ('PI', 'decline_rate')
        ]
        df = engineer.create_interaction_features(df, interaction_pairs)
        
        # Créer la variable cible
        df = engineer.create_target_variable(df, 'well_id', 'date', 'oil_rate')
        
        # Sauvegarder les données avec les nouvelles features
        engineer.save_features(df, 'features_engineered.csv')
        
    except Exception as e:
        logger.error(f"Erreur lors de la création des features: {str(e)}")
        raise

if __name__ == "__main__":
    main() 