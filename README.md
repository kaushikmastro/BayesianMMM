# Bayesian Marketing Mix Model (MMM)

##  Project Overview

This repository contains the necessary components for building, training, and analyzing a sophisticated Bayesian Marketing Mix Model (MMM) using the PyMC framework. The model incorporates advanced features such as Adstock and Seasonality (using Fourier series) to accurately attribute revenue lift to various marketing channels.

The core analysis uses MCMC sampling to derive posterior distributions for key parameters (like channel effectiveness $\beta$ and decay rate $\alpha$), providing robust estimates of Return on Investment (ROI) and allowing for deep probabilistic insight into media performance.

Getting Started

Prerequisites

To run this project, you need a Python environment with the following dependencies installed:

Python (3.9+)

PyMC

ArviZ

Pandas

Matplotlib

You can install all necessary packages using the requirements.txt file (if provided):

pip install -r requirements.txt


📂 Project Structure

The project is organized as follows:

├── my-bayesian-mmm/
│   ├── README.md                 <-- You are here! Project summary, setup, and results.
│   ├── requirements.txt          <-- List of all Python packages.
│   ├── data/
│   │   ├── raw/
│   │   │   └── robyn_synthetic_data.csv    
│   │   └── processed/            <-- For storing normalized and transformed data.
│   ├── src/                      <-- All Python source code.
│   │   ├── utils.py              <-- Helper functions (Normalization, Plotting).
│   │   ├── adstock_functions.py  <-- Adstock transform function.
│   │   └── model_trainer.py      <-- The core 'BayesianMMMTrainer' class.
│   └── notebooks/
│       ├── 01_EDA_Preprocessing.ipynb  
│       └── 02_Model_Results.ipynb      <-- Execution of the Trainer class and analysis.


Saving and Loading the Model

Due to the computational cost of MCMC sampling, it is best practice to save the resulting InferenceData object (the trace) once training is complete. This allows for quick loading and immediate analysis in the future.

```python
import arviz as az
import os

def save_model_results(trace_data, filename="mmm_inference_data.nc"):
    """
    Converts the MCMC trace into an ArviZ InferenceData object and saves it 
    to a NetCDF file for persistent storage.
    """
    print(f"Starting to save inference data to {filename}...")
    try:
        # Use ArviZ's built-in function to save the trace object
        az.to_netcdf(trace_data, filename)
        print(f"Successfully saved model results to {os.path.abspath(filename)}")
    except Exception as e:
        print(f"Error saving file: {e}")

# Example usage (assuming trainer.trace exists):
# save_model_results(trainer.trace)

# To load later:
# loaded_trace = az.from_netcdf("mmm_inference_data.nc")
```

Key Analysis: Model Fit Visualization

After the computationally intensive MCMC sampling is complete, it is crucial to verify how well the model's structural components (base, trend, seasonality, and marketing effects) capture the observed revenue signal.

Purpose: The following code snippet generates the Model Fit Plot. It visualizes the estimated underlying revenue signal ($\mu$) and the uncertainty around the full predicted observations, allowing for a clean assessment of fit against the normalized observed revenue ($y_{obs}$).

Visualization Code 

Model Fit: Observed Revenue vs. Posterior Mean ($\mu$) ---

Get the length of the data used for training

```python
n_weeks = len(trainer.data_df)
```

Generate Posterior Predictive Samples (y_obs is sampled by default)
```python
with trainer.model:
    ppc = pm.sample_posterior_predictive(trainer.trace) 

# Extract predicted observation (y_obs) samples
y_pred_samples = ppc.posterior_predictive['y_obs'].values.reshape(-1, N_WEEKS)
y_obs_norm = trainer.y_revenue_norm.flatten()

# Calculate the Posterior Mean (E[y_obs] ~ mu), which approximates the signal
posterior_mean = np.mean(y_pred_samples, axis=0)

# Plotting the Model Fit with HDI
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the 95% HDI of the full y_obs samples (Shaded Area: Expected range including observation noise)
az.plot_hdi(
    np.arange(N_WEEKS), 
    y_pred_samples, 
    ax=ax, 
    fill_kwargs={'alpha': 0.3, 'label': '95% HDI of Predicted Observations'},
    hdi_prob=0.95
)

# Plot the posterior mean line (Dashed Red Line: The model's cleanest signal estimate)
ax.plot(posterior_mean, color='red', label='Posterior Mean ($\mu$ approximation)', linewidth=2, linestyle='--')

# Plot the observed data (Black Line: Ground Truth)
ax.plot(y_obs_norm, color='black', label='Observed Revenue (Normalized)')

ax.set_title("Model Fit: Observed Revenue vs. Posterior Mean ($\mu$) with 95% HDI")
ax.set_xlabel("Time (Week Index)")
ax.set_ylabel("Revenue (Normalized)")
ax.legend()
plt.show()
```

### Interpretation

If the Black Line (Observed Revenue) falls within the Shaded 95% HDI, the model is correctly capturing the uncertainty and major trends.

The Dashed Red Line ($\mu$) shows the structural fit, indicating the model's estimate of revenue lift without the random observation noise. If this line closely tracks the shape of the Black Line, the model is providing a good behavioral explanation of the data.
