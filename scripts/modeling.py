# -*- coding: utf-8 -*-
"""
__author__ = "Augustine M Gbondo"
__credits__ = ["Augustine M Gbondo"]
__project__ = Geogenic radon potential mapping in Hessen using machine learning techniques
__maintainer__ = "Augustine M Gbondo"
__email__ = "gbondomadaar@gmail.com"
"""

# libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import fiona
import rasterio
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    mean_absolute_percentage_error,root_mean_squared_log_error, mean_squared_log_error)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV, f_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, KFold, LeaveOneGroupOut, 
    cross_validate, cross_val_predict, GridSearchCV)
from sklearn.pipeline import Pipeline


class fit_estimator_with_folds:
    def __init__(self, data, target_column, group_column, model_type='RandomForest',
                 random_state=24, n_estimators=1000, scale_data=False):
        self.data = data
        self.target_column = target_column
        self.group_column = group_column
        self.scale_data = scale_data
        self.model_type = model_type
        self.metrics = {metric: [] for metric in ['RMSE', 'R2', 'MAE', 'MSE','RMSLE', 'R2 Adjusted']}
        self.result_imp = None

        # Scale the data if required
        self.X = data.drop(columns=[target_column, group_column])
        if self.scale_data:
            self.X = self._scaled_data(self.X)

        self.y = data[target_column]
        self.groups = data[group_column]
        self._init_model(random_state, n_estimators)

    def _init_model(self, random_state, n_estimators):
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=n_estimators, max_depth=10,
                                                  oob_score=True, random_state=random_state),
            'GBRT': GradientBoostingRegressor(n_estimators=100, max_depth=None, learning_rate=0.1),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma=0.05),
            'MLPR': MLPRegressor(max_iter=100000, hidden_layer_sizes=(50, 50, 50), activation='tanh',
                                 learning_rate='constant', alpha=0.0001, solver='adam', random_state=random_state)
        }
        if self.model_type in models:
            self.model = models[self.model_type]
        else:
            raise ValueError("Unsupported model_type. Choose from 'RandomForest','GBRT', 'SVR', or 'MLPR'.")

    def _scaled_data(self, unscaled_df, exclude_columns=[]):
        """
        Scale the data excluding specified columns.
        """
        data_to_scale = unscaled_df.copy()
        features_to_scale = [col for col in data_to_scale.columns if col not in exclude_columns]
        scaler = StandardScaler()
        data_to_scale[features_to_scale] = scaler.fit_transform(data_to_scale[features_to_scale])
        return data_to_scale

    def r2_adj(self, X, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        n, p = len(y_true), X.shape[1]
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def train(self):
        loo = LeaveOneGroupOut()
        for train_idx, test_idx in loo.split(self.X, self.y, groups=self.groups):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_train)
            self._store_metrics(X_train, y_train, y_pred)

        # Fit on full data for feature importance
        trained_model = self.model.fit(self.X, self.y)
        self.result_imp = permutation_importance(self.model, self.X, self.y, n_repeats=15, random_state=0)

        return trained_model

    def _store_metrics(self, X, y_true, y_pred):
        self.metrics['RMSE'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
        self.metrics['MSE'].append(mean_squared_error(y_true, y_pred))
        self.metrics['MAE'].append(mean_absolute_error(y_true, y_pred))
        self.metrics['R2'].append(r2_score(y_true, y_pred))
        self.metrics['RMSLE'].append(root_mean_squared_log_error(y_true, y_pred))
        self.metrics['R2 Adjusted'].append(self.r2_adj(X, y_true, y_pred))

    def get_avg_metrics_score(self):
        if not self.metrics['RMSE']:
            self.train()
        return pd.DataFrame(self.metrics).mean().to_frame('Score')

    def get_metrics_df(self):
        if not self.metrics['RMSE']:
            self.train()
        return pd.DataFrame(self.metrics)

    def get_pred_table(self):
        if self.result_imp is None:
            self.train()
        y_pred_full = self.model.predict(self.X)
        return pd.DataFrame({'Actual': self.y, 'Predicted': y_pred_full, 'Residual': self.y - y_pred_full})

    def get_val_scores(self, val_data):
        X_val = val_data.drop(columns=[self.group_column, self.target_column], errors='ignore')
        y_val = val_data[self.target_column]

        # Apply scaling to validation data if required
        if self.scale_data:
            X_val = self._scaled_data(X_val)

        pred_val = self.model.predict(X_val)

        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_val, pred_val)),
            "MSE": mean_squared_error(y_val, pred_val),
            "MAE": mean_absolute_error(y_val, pred_val),
            "R^2": r2_score(y_val, pred_val),
            "RMSLE": root_mean_squared_log_error(y_val, pred_val),
            "Adjusted R^2": self.r2_adj(X_val, y_val, pred_val)
        }

        # Convert metrics to DataFrame with metrics as rows
        all_metrics = pd.DataFrame(metrics, index=[0]).T.rename(columns={0: 'Score'})

        pred_table = pd.DataFrame({
            "Actual": y_val,
            "Predicted": pred_val,
            "Residual": y_val - pred_val
        })

        return pred_table, all_metrics



class Metrics:
    @staticmethod
    def mean_squared_error(true, pred, squared=True):
        """
        Calculates Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) based on the `squared` argument.
        """
        return mean_squared_error(true, pred, squared=squared)

    @staticmethod
    def r2_score(true, pred):
        return r2_score(true, pred)

    @staticmethod
    def mean_absolute_error(true, pred):
        return mean_absolute_error(true, pred)

    @staticmethod
    def mean_absolute_percentage_error(true, pred):
        return mean_absolute_percentage_error(true, pred)

    @staticmethod
    def root_mean_squared_log_error(true, pred):
        """
        Calculates Root Mean Squared Logarithmic Error (RMSLE).
        """
        return root_mean_squared_log_error(true, pred)

    @staticmethod
    def relative_squared_error(true, pred):
        true_mean = np.mean(true)
        squared_error_num = np.sum(np.square(true - pred))
        squared_error_den = np.sum(np.square(true - true_mean))
        return squared_error_num / squared_error_den

    @staticmethod
    def quantile_loss(true, pred, gamma):
        val1 = gamma * np.abs(true - pred)
        val2 = (1 - gamma) * np.abs(true - pred)
        return np.where(true >= pred, val1, val2)

    @staticmethod
    def r2_adj(response, true, pred):
        n = response.shape[0]
        p = response.shape[1]
        r2 = r2_score(true, pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)



def predict_raster(input_stack, model, output_path):

      """
      Predicts raster for each pixels using a trained model.
      
      Parameters:
      - input_stack (str): Path to the input raster stack (multiband raster file).
      - model (object): Trained machine learning model.
      - output_path (str): Output raster save path.

      Returns:
      - None
      """	
      with rasterio.open(input_stack) as src:
        raster_stack = src.read()

        bands, rows, cols = raster_stack.shape
        reshaped_stack = np.moveaxis(raster_stack, 0, -1)
        reshaped_stack = reshaped_stack.reshape(-1, bands)

        predictions = model.predict(reshaped_stack)
        predicted_raster = predictions.reshape(rows, cols)
        meta = src.meta
        meta.update({
            'count': 1,  
            'dtype': 'float32'  
        })
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(predicted_raster, 1)  


def feature_groups(dataframe, feature_groups_list):
    """
    Create a list of feature groups for the given DataFrame, where some features are grouped together.
    """
    filtered_dataframe = dataframe.drop(columns=['GRP', 'grids_36k'], axis=1)
    ungrouped_features = filtered_dataframe.drop(columns=feature_groups_list, axis=1)
    ungrouped_features_list = [
        [filtered_dataframe.columns.get_loc(col)] for col in ungrouped_features]

    grouped_features_tuple = [
        filtered_dataframe.columns.get_loc(col) for col in feature_groups_list
    ]
    grouped_features_list = [grouped_features_tuple]

    # Combine ungrouped and grouped features
    feature_group_all = ungrouped_features_list + grouped_features_list

    return feature_group_all


def aggregate_feature_importance(importance_df, feature_groups_list):
    """
    Aggregate grouped feature importance values.
    """
    # make features index
    fimp_index = importance_df.set_index('Feature')
    aggregated_values = fimp_index.loc[feature_groups_list].sum()

    # Create new row for aggregated values
    new_row = pd.DataFrame({
        'Feature': ['Geology'],
        'Importance': [aggregated_values['Importance']],
        'Permutation Importance': [aggregated_values.get('Permutation Importance', None)]
    })

    # Filter out the grouped features and append the new aggregated row
    importance_df = pd.concat([importance_df[~importance_df['Feature'].isin(feature_groups_list)], new_row], ignore_index=True)
    return importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

def SFS_with_GridSearch(data, target_column, target_variable, group_column, base_estimator, param_grid, feature_groups_list, max_features=35):
    """
    Perform (SFS) with GridSearchCV including feature importances.
    """
    X = data.drop(columns=[target_column, group_column])
    y = target_variable
    groups = data[group_column]
    loo = LeaveOneGroupOut()
    feat_groups = feature_groups(data, feature_groups_list) 

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(base_estimator, param_grid, scoring='r2', cv=loo, verbose=1, n_jobs=-1)
    grid_search.fit(X, y, groups=groups)
    best_estimator = grid_search.best_estimator_

    # Initialize and fit the SFS model with the best estimator
    sfs = SFS(
        best_estimator,
        k_features=max_features,
        forward=True,
        floating=False,
        scoring='r2',
        cv=loo,
        feature_groups=feat_groups,
        verbose=40,
    )

    sfs = sfs.fit(X, y, groups=groups)

    # Plot SFS results
    fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
    plt.title(f'SFS {best_estimator.__class__.__name__} (w. StdErr)')
    plt.grid()
    plt.show()

    # get selected feature, metrics and feature importances
    selected_features = sfs.k_feature_names_
    sfs_metrics = pd.DataFrame.from_dict(sfs.get_metric_dict()).T

    X_selected = X.loc[:, selected_features]
    best_estimator.fit(X_selected, y)

    if hasattr(best_estimator, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': best_estimator.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    else:
        perm_importance = permutation_importance(
            best_estimator, X_selected, y, n_repeats=10, random_state=0)
        feature_importances = pd.DataFrame({
            'Feature': selected_features,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    feature_importances = aggregate_feature_importance(
        feature_importances, feature_groups_list
        )

    return fig, selected_features, sfs_metrics, feature_importances, best_estimator
