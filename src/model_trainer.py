import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler
import arviz as az
import pytensor.tensor as pt
import mlflow
from pathlib import Path

from .adstock_functions import vectorized_geometric_adstock
from .utils import setup_logger, validate_config

logger = setup_logger(__name__)

class BayesianMMMTrainer:
    """
    Production-grade Bayesian Marketing Mix Model (MMM) trainer.

    Orchestrates the complete pipeline: data loading → preprocessing → feature engineering
    (adstock transformations) → Bayesian model training → ROI quantification.

    Leverages PyMC for robust posterior inference and MLflow for reproducible experiment tracking.

    Attributes:
        config: Configuration dictionary with spend_cols, revenue_col, fourier_k, mcmc_params
        data_path: Path to input CSV data
        holidays_path: Optional path to holidays data for control variables
    """
    def __init__(self, config: Dict, data_path: str, holidays_path: Optional[str] = None):
        """Initialize trainer with configuration and data paths.

        Args:
            config: Configuration dict with required keys: date_col, spend_cols, revenue_col,
                   fourier_k, mcmc_params
            data_path: Path to CSV data file
            holidays_path: Optional path to holidays CSV for control variable creation

        Raises:
            ValueError: If configuration is invalid
        """
        validate_config(config)

        self.config = config
        self.data_path = data_path
        self.holidays_path = holidays_path
        self.data_df = None
        self.scalers = {}
        self.trace = None
        self.model = None
        self.data_processed = False

        self.x_spends_norm = None
        self.y_revenue_norm = None
        self.x_seasonality = None
        self.x_trend = None
        self.x_controls = None

        logger.info("BayesianMMMTrainer initialized with validated configuration")
        
# DATA LOADING AND ALIGNMENT #
    
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data: date conversion, grouping, and holiday merging.

        Returns:
            DataFrame with weekly-aggregated data, holidays merged if available

        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If DataFrame is empty after preprocessing
        """
        date_col = self.config['date_col']
        spend_cols = self.config['spend_cols']
        revenue_col = self.config['revenue_col']

        try:
            self.data_df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data from {self.data_path} ({len(self.data_df)} rows)")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Convert Date Column and Set Index
        self.data_df[date_col] = pd.to_datetime(self.data_df[date_col])
        self.data_df = self.data_df.groupby(pd.Grouper(key=date_col, freq='W-MON', label='left')).sum(numeric_only=True)
        self.data_df.index.name = date_col
        self.data_df = self.data_df.sort_index()


        # Load and Merge Holiday Data
        if self.holidays_path:
            try:
                holidays_df = pd.read_csv(self.holidays_path)
                
                # Filter to the relevant country (Assuming Germany 'DE')
                holidays_df = holidays_df[holidays_df['country'] == 'DE'].copy()
                
                # holiday DataFrame: rename, convert date, set index
                holidays_df = holidays_df.rename(columns={'ds': date_col})
                holidays_df[date_col] = pd.to_datetime(holidays_df[date_col])
                
                holidays_df['is_holiday'] = 1
                holidays_df = holidays_df.groupby(pd.Grouper(key=date_col, freq='W-MON', label='left')).max(numeric_only=True)
                holidays_df = holidays_df[['is_holiday']].drop_duplicates()
                holidays_df.index.name = date_col
                
                # Merging based on the index (the date column) to avoid index loss
                self.data_df = self.data_df.merge(
                    holidays_df,
                    left_index=True, # Use index of main data
                    right_index=True, # Use index of holiday data
                    how='left'
                )
                
                self.data_df['is_holiday'] = self.data_df['is_holiday'].fillna(0)  # Fill missing (non-holiday) dates with 0
                
                # Ensure 'is_holiday' is recognized as a control variable
                if 'is_holiday' not in self.config.get('control_cols', []):
                    # Check if control_cols exists; if not, create it
                    if 'control_cols' not in self.config:
                         self.config['control_cols'] = []
                    self.config['control_cols'].append('is_holiday')
                
                logger.info("Holiday data merged successfully")

            except FileNotFoundError:
                logger.warning(f"Holiday file not found at {self.holidays_path}")
            except Exception as e:
                logger.warning(f"Error processing holiday data: {e}")

        # Handle time alignment and fill missing values
        full_index = pd.date_range(start=self.data_df.index.min(), 
                                   end=self.data_df.index.max(), 
                                   freq='W-MON', 
                                   name=date_col) 
        
        # Reindex the data against the full date range.
        self.data_df = self.data_df.reindex(full_index)
        
        # Simple imputation: fill spend and holiday flag with 0
        self.data_df[spend_cols] = self.data_df[spend_cols].fillna(0)
        self.data_df[revenue_col] = self.data_df[revenue_col].ffill() 
        
        # Handle 'is_holiday' column if created, otherwise ensure it's not in control_cols
        if 'is_holiday' in self.data_df.columns:
            self.data_df['is_holiday'] = self.data_df['is_holiday'].fillna(0)
        elif 'is_holiday' in self.config.get('control_cols', []):
            self.config['control_cols'].remove('is_holiday')
            
        self.data_df.dropna(subset=[revenue_col], inplace=True)

        if self.data_df.empty:
            raise ValueError("DataFrame empty after preprocessing. Check data and date ranges")

        logger.info(f"Data loading complete: {len(self.data_df)} weeks")
        return self.data_df

#  DATA PREPROCESSING AND FEATURE GENERATION #
    
    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate features: trend, Fourier seasonality. Normalize revenue and spend.

        Returns:
            Tuple of (x_spends_norm, y_revenue_norm, x_seasonality, x_trend, x_controls)
        """
        if self.data_df is None:
            self.load_data()

        df = self.data_df.copy()
        revenue_col = self.config['revenue_col']
        spend_cols = self.config['spend_cols']
        fourier_k = self.config['fourier_k']
        
        # Feature Generation: Trend
        df['trend'] = np.arange(len(df)) + 1
        self.X_trend = df['trend'].values.reshape(-1, 1)

        # Feature Generation: Fourier Seasonality
        # Generate sine/cosine pairs based on dayofyear
        x_seasonality_list = []
        for k in range(1, fourier_k + 1):
            x_seasonality_list.append(np.sin(2 * k * np.pi * df.index.dayofyear / 365.25))
            x_seasonality_list.append(np.cos(2 * k * np.pi * df.index.dayofyear / 365.25))
            
        self.x_seasonality = np.stack(x_seasonality_list, axis=1)

        # Normalization (Scaling)
        
        # Revenue (Dependent Variable) #
        scaler_y = StandardScaler()
        self.y_revenue_norm = scaler_y.fit_transform(df[[revenue_col]].values)
        self.scalers[revenue_col] = scaler_y
        
        # Spend (Independent Variables) #
        scaler_x = StandardScaler()
        self.x_spends_norm = scaler_x.fit_transform(df[spend_cols].values) # Scale all spend columns using a single scaler to keep track of their relative standard deviations
        self.scalers['spend'] = scaler_x

        # Control Variables #
        control_cols = [col for col in self.config.get('control_cols', []) if col in df.columns]

        if control_cols:
            x_controls_list = []

            for col in control_cols:
                if df[col].nunique() > 2:
                    temp_scaler = StandardScaler()
                    x_scaled = temp_scaler.fit_transform(df[col].values.reshape(-1, 1))
                    self.scalers[f'ctrl_{col}'] = temp_scaler
                    x_controls_list.append(x_scaled)
                else:
                    x_controls_list.append(df[col].values.reshape(-1, 1)) # Binary/dummy controls (like is_holiday) are not scaled

            self.x_controls = np.hstack(x_controls_list)
        else:

            self.x_controls = np.array([[]]).reshape(len(df), 0) 

        self.data_processed = True
        logger.info(f"Preprocessing complete: {p_fourier} Fourier terms, {p_controls} controls")

        return self.x_spends_norm, self.y_revenue_norm, self.x_seasonality, self.x_trend, self.x_controls

    def build_model(self) -> pm.Model:
        """Build Bayesian MMM: adstock + trend + seasonality + control effects.

        Specifies PyMC probabilistic graphical model with:
        - Alpha (decay): Beta priors for adstock decay rates
        - Beta (effectiveness): HalfNormal priors for channel ROI
        - Trend & seasonality: Fourier decomposition for non-marketing effects
        - Controls: Scaling for confounding variables (holidays, etc)

        Returns:
            Compiled PyMC model ready for MCMC sampling
        """
        if not self.data_processed:
            self.preprocess()
            
        n = len(self.data_df)
        p_channels = self.x_spends_norm.shape[1]
        p_fourier = self.x_seasonality.shape[1]
        p_controls = self.x_controls.shape[1]

        coords = {
            "obs_id": np.arange(n),
            "channel": self.config['spend_cols'],
            "fourier_comp": np.arange(p_fourier),
        }
        
      
        if p_controls > 0:
            control_names = [c for c in self.config.get('control_cols', []) if c in self.data_df.columns]
            coords["control_comp"] = control_names
        

        with pm.Model(coords=coords) as self.model:
            
            # Shared Data (for MCMC and future prediction)
            x_spends_shared = pm.MutableData("x_spends_norm", self.x_spends_norm)
            x_seasonality_shared = pm.MutableData("x_seasonality", self.x_seasonality)
            x_trend_shared = pm.MutableData("x_trend", self.x_trend)
            
            # Only include control variables if they exist
            if p_controls > 0:
                x_controls_shared = pm.MutableData("x_controls", self.x_controls)

            # Priors
            alpha = pm.Beta("alpha", 2, 8, dims="channel")
            beta = pm.HalfNormal("beta", sigma=1, dims="channel")
            intercept = pm.Normal("intercept", mu=0, sigma=10)
            trend_coef = pm.Normal("trend_coef", mu=0, sigma=1)
            seasonality_weights = pm.Normal("seasonality_weights", mu=0, sigma=1, dims="fourier_comp")

            # Control Priors and Effect Calculation
            if p_controls > 0:
                control_coefs = pm.Normal("control_coefs", mu=0, sigma=1, dims="control_comp")
                control_effect = pm.math.dot(x_controls_shared, control_coefs)
            else:
                control_effect = pt.constant(0.0)

            # Error term prior
            sigma = pm.HalfCauchy("sigma", beta=1)
            x_adstock = vectorized_geometric_adstock(x_spends_shared, alpha)
            media_effect = pm.math.dot(x_adstock, beta)
            
            # Baseline Components
            trend_effect = trend_coef * x_trend_shared[:, 0]
            seasonality_effect = pm.math.dot(x_seasonality_shared, seasonality_weights)

            # Full Model Mean
            mu = intercept + media_effect + trend_effect + seasonality_effect + control_effect

            # Observed Normalized Revenue (The Likelihood Function)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y_revenue_norm[:, 0], dims="obs_id") 

        logger.info(f"Model built: {p_channels} channels, {p_fourier} Fourier terms, {p_controls} controls")
        return self.model
        
#  SAMPLING AND ANALYSIS #
    
    def train(self, experiment_name: str = "mmm_baseline") -> az.InferenceData:
        """Execute MCMC sampling to estimate posterior distribution.

        Uses NUTS sampler for efficient exploration of parameter space. Validates
        convergence via R-hat diagnostics (target: <1.05).

        Args:
            experiment_name: MLflow experiment name for tracking

        Returns:
            ArviZ InferenceData object with posterior samples and diagnostics

        Raises:
            RuntimeError: If MCMC sampling fails
        """
        if self.model is None:
            self.build_model()

        mcmc_params = self.config['mcmc_params']

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            logger.info(f"Starting MCMC: draws={mcmc_params['draws']}, tune={mcmc_params['tune']}")

            with self.model:
                self.trace = pm.sample(
                    draws=mcmc_params['draws'],
                    tune=mcmc_params['tune'],
                    target_accept=mcmc_params['target_accept'],
                    return_inferencedata=True,
                    progressbar=True
                )

            summary = az.summary(self.trace, var_names=["beta", "alpha", "sigma"])

            if (summary['r_hat'] > 1.05).any():
                logger.warning("High R-hat values detected (>1.05). Check convergence.")
            else:
                logger.info("All R-hat values < 1.05. Model converged successfully.")

            # Log metrics to MLflow
            mlflow.log_param("n_draws", mcmc_params['draws'])
            mlflow.log_param("n_tune", mcmc_params['tune'])
            mlflow.log_param("target_accept", mcmc_params['target_accept'])
            mlflow.log_metric("mean_r_hat", summary['r_hat'].mean())

            logger.info("Sampling complete")

        return self.trace

    def calculate_roi(self) -> Dict[str, Dict[str, float]]:
        """Calculate unscaled ROI per channel from posterior estimates.

        ROI = beta * (1 / (1 - alpha)) * (sigma_y / sigma_x)

        Where:
        - beta: posterior mean channel effectiveness
        - alpha: posterior mean adstock decay rate
        - sigma_y, sigma_x: standard deviations from training normalization

        Returns:
            Dict mapping channel names to {mean_beta, mean_alpha, unscaled_roi}

        Raises:
            ValueError: If model has not been trained
        """
        if self.trace is None:
            raise ValueError("Train model before calculating ROI")

        beta_post = self.trace.posterior["beta"].mean(dim=["chain", "draw"]).values
        alpha_post = self.trace.posterior["alpha"].mean(dim=["chain", "draw"]).values

        revenue_col = self.config['revenue_col']
        revenue_std = self.scalers[revenue_col].scale_[0]
        spend_std = self.scalers['spend'].scale_

        roi_results = {}

        for i, channel in enumerate(self.config['spend_cols']):
            alpha = alpha_post[i]
            beta = beta_post[i]
            sigma_x_i = spend_std[i]

            lifetime_multiplier = 1 / (1 - alpha)
            unscaling_factor = revenue_std / sigma_x_i
            roi = beta * lifetime_multiplier * unscaling_factor

            roi_results[channel] = {
                "mean_beta": float(beta),
                "mean_alpha": float(alpha),
                "unscaled_roi": float(roi)
            }

            logger.info(f"{channel:20s} | ROI: {roi:8.2f} | Beta: {beta:6.3f} | Alpha: {alpha:6.3f}")

        return roi_results

        
    def save_trace(self, filepath: str) -> None:
        """Persist MCMC trace to NetCDF for later analysis."""
        if self.trace is None:
            raise ValueError("No trace to save. Train model first.")
        az.to_netcdf(self.trace, filepath)
        logger.info(f"Trace saved to {filepath}")

    def load_trace(self, filepath: str) -> az.InferenceData:
        """Load previously saved MCMC trace."""
        self.trace = az.from_netcdf(filepath)
        logger.info(f"Trace loaded from {filepath}")
        return self.trace

    def run_full_analysis(self, experiment_name: str = "mmm_baseline") -> Dict[str, Dict[str, float]]:
        """Execute complete MMM pipeline: data → model → inference → ROI.

        Args:
            experiment_name: MLflow experiment name

        Returns:
            Dictionary of channel ROI results
        """
        logger.info("=== Starting Full MMM Pipeline ===")
        self.load_data()
        self.preprocess()
        self.build_model()
        self.train(experiment_name=experiment_name)
        roi = self.calculate_roi()
        logger.info("=== Pipeline Complete ===")
        return roi
