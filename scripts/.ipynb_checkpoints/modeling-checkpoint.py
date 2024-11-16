import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    mean_absolute_percentage_error, mean_squared_log_error, 
    root_mean_squared_log_error)


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


from sklearn.model_selection import LeaveOneGroupOut
from sklearn.inspection import permutation_importance



class MultiModelLOO:
    def __init__(self, data, target_column, group_column, model_type='RandomForest', random_state=24, n_estimators=1000):
        self.data = data
        self.target_column = target_column
        self.group_column = group_column
        self.X = data.drop(columns=[target_column, group_column])
        self.y = data[target_column]
        self.groups = data[group_column]
        self.model_type = model_type
        self.metrics = {metric: [] for metric in ['RMSE', 'R2', 'MAE', 'MSE',
                                                  'MAPE', 'RMSLE', 'MBE', 'RRSE', 'R2 Adjusted']}
        self.result_imp = None
        self._init_model(random_state, n_estimators)

    def _init_model(self, random_state, n_estimators):
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=n_estimators, max_depth=10,
                                                  oob_score=True, random_state=random_state),
            'GBRT': GradientBoostingRegressor(n_estimators=100, max_depth=None, learning_rate=0.1), #GradientBoostingRegressor(n_estimators=n_estimators, max_depth=7, learning_rate=0.01, random_state=random_state),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma = 0.05), #SVR(kernel='rbf',C= 1, epsilon= 1, gamma= 0.0001),
            'MLPR': MLPRegressor(max_iter= 100000,   hidden_layer_sizes=(50,50,50), activation='tanh',
                                 learning_rate='constant', alpha=0.0001, solver='adam',random_state=random_state)
        }
        if self.model_type in models:
            self.model = models[self.model_type]
        else:
            raise ValueError("Unsupported model_type. Choose from 'RandomForest','GBRT', 'SVR', or 'MLPR'.")

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
        self.metrics['MAPE'].append(mean_absolute_percentage_error(y_true, y_pred))
        self.metrics['RMSLE'].append(root_mean_squared_log_error(y_true, y_pred))
        self.metrics['MBE'].append(np.mean(y_true - y_pred))
        self.metrics['RRSE'].append(np.sqrt(np.mean((y_true - y_pred)**2) / np.mean((y_true - np.mean(y_true))**2)))
        self.metrics['R2 Adjusted'].append(Metrics.r2_adj(X, y_true, y_pred))

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

    def get_feature_importance(self, feature_groups_list):
        if self.result_imp is None:
            self.train()

        # Compute feature importances
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Model Importance': self.model.feature_importances_,
            'Permutation Importance': self.result_imp.importances_mean
        })

        # Aggregate grouped feature importance
        fimp_index = importance_df.set_index('Feature')
        aggregated_values = fimp_index.loc[feature_groups_list].sum()
        new_row = pd.DataFrame({
            'Feature': 'Geology',
            'Model Importance': [aggregated_values['Model Importance']],
            'Permutation Importance': [aggregated_values['Permutation Importance']]
        })
        importance_df = pd.concat([importance_df[~importance_df['Feature'].isin(feature_groups_list)], new_row], ignore_index=True)
        return importance_df.sort_values(by='Model Importance', ascending=False).reset_index(drop=True)

    def predict(self, data):
        X_new = data.drop(columns=[self.target_column, self.group_column], errors='ignore')
        return self.model.predict(X_new)

    def get_val_scores(self, val_data):
        X_val = val_data.drop(columns=[self.group_column, self.target_column], errors='ignore')
        y_val = val_data[self.target_column]
        pred_val = self.model.predict(X_val)

        metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_val, pred_val)),
            "MSE": mean_squared_error(y_val, pred_val),
            "MAE": mean_absolute_error(y_val, pred_val),
            "R^2": r2_score(y_val, pred_val),
            "MAPE": mean_absolute_percentage_error(y_val, pred_val),
            "RMSLE": root_mean_squared_log_error(y_val, pred_val),
            "MBE": np.mean(y_val - pred_val),
            "RRSE": np.sqrt(np.mean((y_val - pred_val)**2) / np.mean((y_val - np.mean(y_val))**2)),
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

