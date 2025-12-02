import pandas as pd
from typing import Dict, List, Optional
import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler
import arviz as az
import pytensor.tensor as pt
import logging
import os

from .adstock_functions import vectorized_geometric_adstock


logging.basicConfig(level=logging.INFO)

class BayesianMMMTrainer:
    """
    The core tool for running a Bayesian Marketing Mix Model (MMM).
    Handles data loading, preprocessing, model building, MCMC sampling, and analysis.
    """
    def __init__(self, config: Dict, data_path: str, holidays_path: Optional[str] = None):
        """
        Initializes the trainer with file paths and configuration.
        """
        self.config = config
        self.data_path = data_path
        self.holidays_path = holidays_path
        self.data_df = None       # Stores the merged and preprocessed DataFrame
        self.scalers = {}         # Stores scalers (StandardScaler objects) for unscaling ROI
        self.trace = None         # Stores the MCMC posterior trace (az.InferenceData)
        self.model = None         # Stores the PyMC model object
        self.data_processed = False
        
        # Data preparation matrices (to be set in preprocess)
        self.x_spends_norm = None
        self.y_revenue_norm = None
        self.x_seasonality = None
        self.x_trend = None
        self.x_controls = None
        
        print("Trainer initialized. Configuration loaded.")
        
# DATA LOADING AND ALIGNMENT #
    
    def load_data(self):
        """
        Loads the main data, converts the date index, and merges holiday data.
        Handles time alignment and simple imputation.
        """
        date_col = self.config['date_col']
        spend_cols = self.config['spend_cols']
        revenue_col = self.config['revenue_col']
        
        # Load Main Data
        try:
            self.data_df = pd.read_csv(self.data_path)
            print(f"Loaded main data from: {self.data_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Main data file not found at {self.data_path}")

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
                
                # Merge based on the index (the date column) to avoid index loss
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
                
                print(f"Merged and filtered holiday data successfully.")
            
            except FileNotFoundError:
                print(f"Warning: Holiday file not found at {self.holidays_path}. Skipping merge.")
            except Exception as e:
                print(f"Warning: Error processing holiday data: {e}. Skipping merge.")

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

        print(f"Data loading and initial alignment complete. Total weeks: {len(self.data_df)}")
        
        # Check if data is empty after cleanup, which would cause the StandardScaler error
        if self.data_df.empty:
            raise ValueError("Data frame is empty after alignment and imputation. Check your simulated data and date ranges.")
            
        return self.data_df

#  DATA PREPROCESSING AND FEATURE GENERATION #
    
    def preprocess(self):
        """
        Generates Trend, Fourier Seasonality, and scales all necessary variables.
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
            X_seasonality_list.append(np.sin(2 * k * np.pi * df.index.dayofyear / 365.25))
            X_seasonality_list.append(np.cos(2 * k * np.pi * df.index.dayofyear / 365.25))
            
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
                    x_controls_list.append(X_scaled)
                else: 
                    # Binary/dummy controls (like is_holiday) are not scaled
                    X_controls_list.append(df[col].values.reshape(-1, 1))

            self.x_controls = np.hstack(X_controls_list)
        else:
            
             self.x_controls = np.array([[]]).reshape(len(df), 0) 

        self.data_processed = True
        print("Data preprocessing complete: Trend, Seasonality, and Scaling applied.")
        
        return self.x_spends_norm, self.y_revenue_norm, self.x_seasonality, self.x_trend, self.x_controls

    # MODEL BUILDING #
    def build_model(self):
        """
        Defines the Bayesian MMM structure using PyMC.
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
            alpha = pm.Beta("alpha", 2, 8, dims="channel")# Alpha: Adstock decay rate (Beta distribution often used for [0, 1] range)        
           
            beta = pm.HalfNormal("beta", sigma=1, dims="channel")  # Beta: Channel effectiveness (HalfNormal for positive effectiveness)   
            
            intercept = pm.Normal("intercept", mu=0, sigma=10)      
            
            trend_coef = pm.Normal("trend_coef", mu=0, sigma=1)
            
            seasonality_weights = pm.Normal("seasonality_weights", mu=0, sigma=1, dims="fourier_comp")
            
            # Control Priors and Effect Calculation
            if p_controls > 0:
                control_coefs = pm.Normal("control_coefs", mu=0, sigma=1, dims="control_comp")
                control_effect = pm.math.dot(X_controls_shared, control_coefs)
                print(f"DEBUG: Control effect is a PyTensor variable (P_controls={P_controls})")
            else:
        
                control_effect = pt.constant(0.0) 
                print(f"DEBUG: Control effect is a PyTensor constant 0.0 (P_controls={P_controls})")

            # Error term prior
            sigma = pm.HalfCauchy("sigma", beta=1)
            X_adstock = vectorized_geometric_adstock(x_spends_shared, alpha) 
            media_effect = pm.math.dot(x_adstock, beta)
            
            # Baseline Components
            trend_effect = trend_coef * x_trend_shared[:, 0]
            seasonality_effect = pm.math.dot(x_seasonality_shared, seasonality_weights)

            # Full Model Mean
            mu = intercept + media_effect + trend_effect + seasonality_effect + control_effect

            # Observed Normalized Revenue (The Likelihood Function)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y_revenue_norm[:, 0], dims="obs_id") 

        print(f"Model built successfully with {P_channels} channels and {P_controls} control features.")
        return self.model
        
#  SAMPLING AND ANALYSIS #
    
    def train(self):
        """
        Samples the posterior distribution using NUTS.
        """
        if self.model is None:
            self.build_model()
            
        mcmc_params = self.config['mcmc_params']
        
        print(f"Starting MCMC sampling: Draws={mcmc_params['draws']}, Tune={mcmc_params['tune']}")
        
        with self.model:
            self.trace = pm.sample(
                draws=mcmc_params['draws'], 
                tune=mcmc_params['tune'], 
                target_accept=mcmc_params['target_accept'], 
                return_inferencedata=True
            )
        
        # Basic Convergence Check
        summary = az.summary(self.trace, var_names=["beta", "alpha", "sigma"])
        if (summary['r_hat'] > 1.05).any():
            print("\nWARNING: Some R-hat values are high (>1.05). Check convergence.")
        
        print("\nSampling complete. Posterior Summary (Partial):")
        print(summary)
        return self.trace

    def calculate_roi(self):
        """
        Calculates the unscaled ROI for each channel using posterior means and scalers.
        """
        if self.trace is None:
            raise ValueError("Model must be trained before calculating ROI.")
            
        # Extract Posterior Means
        beta_post = self.trace.posterior["beta"].mean(dim=["chain","draw"]).values
        alpha_post = self.trace.posterior["alpha"].mean(dim=["chain","draw"]).values
        
        # Extract Unscaling Factors
        revenue_col = self.config['revenue_col']
        revenue_std = self.scalers[revenue_col].scale_[0] 
        # Assume 'spend' scaler for all spend columns
        spend_std = self.scalers['spend'].scale_ 
        
        roi_results = {}
        
        # Calculate ROI for Each Channel
        for i, channel in enumerate(self.config['spend_cols']):
            alpha = alpha_post[i]
            beta = beta_post[i]
            sigma_x_i = spend_std[i]
            
            lifetime_multiplier = 1 / (1 - alpha)  # ROI Formula: (beta * Lifetime Multiplier) * Unscaling Factor
            
            unscaling_factor = revenue_std / sigma_x_i # Unscaling Factor: sigma_y / sigma_x_i
            
            roi = beta * lifetime_multiplier * unscaling_factor
            
            roi_results[channel] = {
                "mean_beta": beta,
                "mean_alpha": alpha,
                "unscaled_roi": roi
            }
            
            print(f"ROI for {channel}: {roi:.2f}")

        return roi_results

        
    def run_full_analysis(self):
        """Convenience method to run the entire pipeline."""
        print("--- STARTING FULL MMM PIPELINE ---")
        self.load_data()
        self.preprocess()
        self.build_model()
        self.train()
        roi = self.calculate_roi()
        self.plot_ppc()
        print("--- PIPELINE COMPLETE ---")
        return roi
