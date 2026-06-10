import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# Import our previously built data pipeline
from scripts.features_engineering import FeatureEngineer
from scripts.model_selection import CrossValidator


class ModelTrainer:
    """
    Constructs an AutoML pipeline, evaluates multiple models over custom time-series folds,
    and extracts feature importances.
    """

    def __init__(self, df: pd.DataFrame, cv_folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.df = df.reset_index(drop=True)
        self.date_folds = cv_folds
        
        self.features = ['open', 'high', 'low', 'close', 'volume', 'bb_high', 'bb_low', 'rsi', 'macd', 'macd_signal']
        self.target = 'target'
        
        # XGBoost requires targets to be strictly >= 0.
        # We shift our targets mathematically: -1 -> 0,  0 -> 1,  1 -> 2
        logger.info("Shifting target labels by +1 to satisfy XGBoost strict indexing (0, 1, 2)...")
        self.df[self.target] = self.df[self.target] + 1
        
        self.cv_indices: list[tuple[list[int], list[int]]] = []
        self._map_dates_to_indices()

    def _map_dates_to_indices(self) -> None:
        """Converts the date-based folds into strict row-index arrays."""
        try:
            for train_dates, val_dates in self.date_folds:
                train_idx = self.df.index[self.df['date'].isin(train_dates)].tolist()
                val_idx = self.df.index[self.df['date'].isin(val_dates)].tolist()
                self.cv_indices.append((train_idx, val_idx))
        except Exception as e:
            logger.error(f"Failed to map CV indices: {e}")
            raise

    def build_pipeline(self, model_instance) -> Pipeline:
        """Constructs the Pipeline with a dynamic model injection."""
        try:
            # We keep Random Forest internal to RFE for feature ranking
            rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('rfe', RFE(estimator=rfe_estimator, n_features_to_select=7)),
                ('model', model_instance)
            ])
            return pipeline
        except Exception as e:
            logger.error(f"Failed to build pipeline: {e}")
            raise

    def execute_grid_search(self, pipeline: Pipeline, param_grid: dict) -> GridSearchCV:
        """Executes parallelized GridSearchCV optimizing for ROC AUC."""
        try:
            X = self.df[self.features]
            y = self.df[self.target]

            # n_jobs=-1 uses all cores. scoring='roc_auc_ovr' prevents the Accuracy Trap.
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=self.cv_indices,
                scoring='roc_auc_ovr',
                verbose=1,
                n_jobs=-1 
            )
            
            grid_search.fit(X, y)
            logger.info(f"Best parameters for this model: {grid_search.best_params_}")
            return grid_search
        except Exception as e:
            logger.error(f"Failed during GridSearch: {e}")
            raise

    def evaluate_and_extract(self, best_pipeline: Pipeline, save_dir: str = "results/cross-validation") -> None:
        """Loops through the CV folds with the ultimate best model to extract metrics and importance."""
        os.makedirs(save_dir, exist_ok=True)
        metrics_list = []
        feature_importance_list = []
        
        try:
            for fold, (train_idx, val_idx) in enumerate(self.cv_indices):
                X_train, y_train = self.df.loc[train_idx, self.features], self.df.loc[train_idx, self.target]
                X_val, y_val = self.df.loc[val_idx, self.features], self.df.loc[val_idx, self.target]

                best_pipeline.fit(X_train, y_train)

                y_train_pred = best_pipeline.predict(X_train)
                y_val_pred = best_pipeline.predict(X_val)
                y_train_proba = best_pipeline.predict_proba(X_train)
                y_val_proba = best_pipeline.predict_proba(X_val)

                metrics = {
                    'Fold': fold + 1,
                    'Train_AUC': roc_auc_score(y_train, y_train_proba, multi_class='ovr'),
                    'Val_AUC': roc_auc_score(y_val, y_val_proba, multi_class='ovr'),
                    'Train_LogLoss': log_loss(y_train, y_train_proba),
                    'Val_LogLoss': log_loss(y_val, y_val_proba)
                }
                metrics_list.append(metrics)

                # Extract Feature Importance (Handling models without feature_importances_ like Logistic Regression)
                surviving_mask = best_pipeline.named_steps['rfe'].get_support()
                surviving_features = np.array(self.features)[surviving_mask]
                
                final_model = best_pipeline.named_steps['model']
                if hasattr(final_model, 'feature_importances_'):
                    importances = final_model.feature_importances_
                elif hasattr(final_model, 'coef_'):
                    # For Logistic Regression, we average the absolute coefficients across classes
                    importances = np.mean(np.abs(final_model.coef_), axis=0)
                else:
                    importances = np.zeros(len(surviving_features))
                
                feat_df = pd.DataFrame({'Feature': surviving_features, 'Importance': importances})
                feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
                feat_df['Fold'] = fold + 1
                feature_importance_list.append(feat_df)

            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(os.path.join(save_dir, "ml_metrics_train.csv"), index=False)
            
            fi_df = pd.concat(feature_importance_list, ignore_index=True)
            fi_df.to_csv(os.path.join(save_dir, "top_10_feature_importance.csv"), index=False)

            plt.figure(figsize=(10, 6))
            plt.plot(metrics_df['Fold'], metrics_df['Train_AUC'], label='Train AUC', marker='o')
            plt.plot(metrics_df['Fold'], metrics_df['Val_AUC'], label='Validation AUC', marker='s')
            plt.title('AUC Stability Across Chronological Folds')
            plt.xlabel('Time-Series Fold')
            plt.ylabel('ROC AUC Score (OVR)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "metric_train.png"))
            plt.close()

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def save_model(self, grid_search: GridSearchCV, save_dir: str = "results/selected-model") -> None:
        """Saves the ultimate winning pipeline to disk."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(grid_search.best_estimator_, os.path.join(save_dir, "selected_model.pkl"))
            with open(os.path.join(save_dir, "selected_model.txt"), "w") as f:
                f.write("Ultimate Best Hyperparameters:\n")
                for key, val in grid_search.best_params_.items():
                    f.write(f"{key}: {val}\n")
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise


if __name__ == "__main__":
    try:
        engineer = FeatureEngineer(index_data_path="data/HistoricalData.csv", constituents_data_path="data/all_stocks_5yr.csv")
        engineer.load_data()
        engineer.define_target()
        engineer.apply_technical_indicators()
        train_df, _ = engineer.split_train_test()

        cv_engine = CrossValidator(n_splits=10)
        cv_engine.generate_splits(train_df)

        trainer = ModelTrainer(df=train_df, cv_folds=cv_engine.date_folds)
        
        # --- The AutoML Contestants ---
        candidate_models = {
            "LogisticRegression": {
                # Simple linear model to test if the market regime is just a trend
                "model": LogisticRegression(random_state=42, multi_class='ovr', max_iter=1000, n_jobs=1),
                "params": {'model__C': [0.1, 1.0]}
            },
            "RandomForest": {
                # Bagging model to capture complex logic
                "model": RandomForestClassifier(random_state=42, n_jobs=1),
                "params": {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]}
            },
            "XGBoost": {
                # Boosting model for aggressive error correction
                "model": XGBClassifier(random_state=42, objective='multi:softprob', eval_metric='mlogloss', n_jobs=1),
                "params": {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 5]}
            }
        }

        best_overall_model = None
        best_overall_score = 0

        # --- The Evaluation Loop ---
        for name, config in candidate_models.items():
            logger.info(f"====== Evaluating {name} ======")
            pipeline = trainer.build_pipeline(model_instance=config["model"])
            grid_result = trainer.execute_grid_search(pipeline, param_grid=config["params"])
            
            logger.info(f"{name} Best AUC: {grid_result.best_score_:.4f}")
            
            if grid_result.best_score_ > best_overall_score:
                best_overall_score = grid_result.best_score_
                best_overall_model = grid_result

        logger.success(f"🏆 ULTIMATE WINNER: {best_overall_model.best_estimator_.named_steps['model'].__class__.__name__} with AUC: {best_overall_score:.4f} 🏆")
        
        best_pipe = best_overall_model.best_estimator_
        trainer.evaluate_and_extract(best_pipeline=best_pipe)
        trainer.save_model(grid_search=best_overall_model)
        
    except Exception as main_e:
        logger.critical(f"Pipeline execution failed: {main_e}")