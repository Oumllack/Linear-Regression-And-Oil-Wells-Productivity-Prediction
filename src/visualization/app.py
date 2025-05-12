"""
Dashboard interactif pour la visualisation des prédictions de productivité des puits.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Optional
import shap

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Chargement des données et des modèles
def load_data_and_models():
    """Charge les données et les modèles entraînés."""
    try:
        # Charger les données
        data_path = Path('data/processed/features_engineered.csv')
        df = pd.read_csv(data_path)
        
        # Charger les modèles
        models_path = Path('models')
        xgb_model = joblib.load(models_path / 'xgboost.joblib')
        rf_model = joblib.load(models_path / 'random_forest.joblib')
        
        # Charger le scaler
        scaler = joblib.load(models_path / 'feature_scaler.joblib')
        
        # Charger l'importance des features
        xgb_importance = pd.read_csv(models_path / 'xgboost_importance.csv')
        rf_importance = pd.read_csv(models_path / 'random_forest_importance.csv')
        
        return {
            'data': df,
            'models': {
                'xgboost': xgb_model,
                'random_forest': rf_model
            },
            'scaler': scaler,
            'feature_importance': {
                'xgboost': xgb_importance,
                'random_forest': rf_importance
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données et modèles: {str(e)}")
        raise

# Chargement initial
data_dict = load_data_and_models()
df = data_dict['data']
models = data_dict['models']
scaler = data_dict['scaler']
feature_importance = data_dict['feature_importance']

# Layout de l'application
app.layout = html.Div([
    # En-tête
    html.H1("Dashboard de Prédiction de Productivité des Puits",
            style={'textAlign': 'center', 'color': '#2c3e50', 'margin': '20px'}),
    
    # Conteneur principal
    html.Div([
        # Panneau de contrôle
        html.Div([
            html.H3("Paramètres", style={'color': '#2c3e50'}),
            
            # Sélection du puits
            html.Label("Sélectionner un puits:"),
            dcc.Dropdown(
                id='well-dropdown',
                options=[{'label': well, 'value': well} for well in df['well_id'].unique()],
                value=df['well_id'].iloc[0]
            ),
            
            # Sélection du modèle
            html.Label("Sélectionner un modèle:"),
            dcc.RadioItems(
                id='model-radio',
                options=[
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'Random Forest', 'value': 'random_forest'}
                ],
                value='xgboost',
                labelStyle={'display': 'block', 'margin': '10px 0'}
            ),
            
            # Bouton de prédiction
            html.Button('Générer Prédiction', id='predict-button',
                       style={'margin': '20px 0', 'padding': '10px 20px',
                             'backgroundColor': '#3498db', 'color': 'white',
                             'border': 'none', 'borderRadius': '5px'})
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                 'borderRadius': '10px', 'margin': '10px'}),
        
        # Visualisations
        html.Div([
            # Graphique de production
            dcc.Graph(id='production-plot'),
            
            # Graphique d'importance des features
            dcc.Graph(id='feature-importance-plot'),
            
            # Graphique SHAP
            dcc.Graph(id='shap-plot'),
            
            # Métriques de performance
            html.Div(id='metrics-display', style={'margin': '20px'})
        ], style={'width': '75%', 'padding': '20px'})
    ], style={'display': 'flex'})
])

# Callbacks
@app.callback(
    [Output('production-plot', 'figure'),
     Output('feature-importance-plot', 'figure'),
     Output('shap-plot', 'figure'),
     Output('metrics-display', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('well-dropdown', 'value'),
     State('model-radio', 'value')]
)
def update_plots(n_clicks, selected_well, selected_model):
    """Met à jour les visualisations en fonction des sélections."""
    if n_clicks is None:
        # Retourner des figures vides au chargement initial
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Sélectionnez un puits et cliquez sur 'Générer Prédiction'")
        return empty_fig, empty_fig, empty_fig, ""
    
    try:
        # Filtrer les données pour le puits sélectionné
        well_data = df[df['well_id'] == selected_well].copy()
        
        # Préparer les features pour la prédiction
        feature_cols = [
            'oil_rate', 'bottomhole_pressure', 'water_cut',
            'decline_rate', 'PI', 'cumulative_production',
            'water_cut_trend', 'year', 'month'
        ]
        X = well_data[feature_cols]
        X_scaled = scaler.transform(X)
        
        # Faire la prédiction
        model = models[selected_model]
        predictions = model.predict(X_scaled)
        
        # Créer le graphique de production
        production_fig = go.Figure()
        production_fig.add_trace(go.Scatter(
            x=well_data['date'],
            y=well_data['oil_rate'],
            name='Production Réelle',
            line=dict(color='#2ecc71')
        ))
        production_fig.add_trace(go.Scatter(
            x=well_data['date'],
            y=predictions,
            name='Prédiction',
            line=dict(color='#e74c3c', dash='dash')
        ))
        production_fig.update_layout(
            title=f"Production vs Prédiction - Puits {selected_well}",
            xaxis_title="Date",
            yaxis_title="Taux de Production (bbl/jour)",
            template="plotly_white"
        )
        
        # Créer le graphique d'importance des features
        importance_data = feature_importance[selected_model]
        importance_fig = px.bar(
            importance_data,
            x='importance',
            y='feature',
            orientation='h',
            title=f"Importance des Features - {selected_model.capitalize()}",
            template="plotly_white"
        )
        importance_fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        # Créer le graphique SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled)
        
        # Créer un graphique de résumé SHAP
        shap_fig = go.Figure()
        for i, feature in enumerate(feature_cols):
            shap_fig.add_trace(go.Scatter(
                x=X_scaled[:, i],
                y=shap_values.values[:, i],
                mode='markers',
                name=feature,
                marker=dict(
                    size=8,
                    color=shap_values.values[:, i],
                    colorscale='RdBu',
                    showscale=True
                )
            ))
        shap_fig.update_layout(
            title="Valeurs SHAP par Feature",
            xaxis_title="Valeur Feature (normalisée)",
            yaxis_title="Valeur SHAP",
            template="plotly_white"
        )
        
        # Calculer les métriques
        mse = np.mean((well_data['oil_rate'] - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(well_data['oil_rate'] - predictions))
        r2 = 1 - mse / np.var(well_data['oil_rate'])
        
        metrics_display = html.Div([
            html.H4("Métriques de Performance", style={'color': '#2c3e50'}),
            html.Table([
                html.Tr([html.Td("MSE"), html.Td(f"{mse:.2f}")]),
                html.Tr([html.Td("RMSE"), html.Td(f"{rmse:.2f}")]),
                html.Tr([html.Td("MAE"), html.Td(f"{mae:.2f}")]),
                html.Tr([html.Td("R²"), html.Td(f"{r2:.2f}")])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        ])
        
        return production_fig, importance_fig, shap_fig, metrics_display
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des visualisations: {str(e)}")
        raise

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 