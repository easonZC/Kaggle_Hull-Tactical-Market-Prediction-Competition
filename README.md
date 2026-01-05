# Hull Tactical Market Prediction (HTMP)

This repository contains the advanced machine learning pipeline used for the Hull Tactical Market Prediction Competition. The system employs a sophisticated ensemble of Gradient Boosting (CatBoost) and Deep Learning (GRU, Transformer) models, enhanced by Genetic Programming for feature engineering.

## Key Features

- **Advanced Feature Engineering**: 
    - Automated generation of rolling statistics, lag features, and cross-sectional z-scores.
    - **Symbolic Transformer (Genetic Programming)**: Uses `gplearn` to mathematically evolve new nonlinear features that maximize correlation with the target.
- **Hybrid Ensemble Architecture**:
    - **CatBoost**: Handles tabular data and categorical interactions efficiently.
    - **GRU (Gated Recurrent Unit)**: Captures sequential dependencies and temporal patterns.
    - **Transformer**: Utilizes self-attention mechanisms to identify long-range dependencies.
- **Robust Validation**:
    - Implements **Time-Series Cross-Validation** to prevent look-ahead bias.
    - Uses a custom **"Mining vs. Evaluation"** split scheme to separate feature selection from model performance scoring, reducing overfitting.
- **Custom Optimization Metric**:
    - Models are optimized not just for MSE, but for a custom **Adjusted Sharpe Ratio** that penalizes excessive volatility and rewards consistent returns relative to the market.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hull-tactical-market-prediction.git
   cd hull-tactical-market-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration & Usage

The main script `hull_tactical_market_prediction.py` is designed to be configurable. **You must set the strategy parameters before running.**

1. Open `hull_tactical_market_prediction.py`.
2. Locate the `GLOBAL CONFIGURATION` section.
3. Update the following parameters with values appropriate for your dataset:

   ```python
   # Example Configuration
   DATA_DIR = "./data"             # Path to your CSV files
   MINE_FRAC = 0.60                # Use first 60% of data for feature mining
   TOP_K_FEATURES = 150            # Select top 150 features
   ENSEMBLE_W = (0.4, 0.4, 0.2)    # Weights for CatBoost, GRU, Transformer
   ALPHA = 100.0                   # Aggressiveness of position scaling
   ```

4. Run the training pipeline:
   ```bash
   python hull_tactical_market_prediction.py
   ```

## Pipeline Artifacts

The script will generate artifacts in the `./artifacts` directory, including:
- `cat_model.cbm`: Trained CatBoost model.
- `gru_model.pth` / `trf_model.pth`: PyTorch model weights.
- `symbolic_model.pkl`: The evolved genetic feature transformer.
- `meta.npz`: Metadata including feature names, scaling stats, and ensemble weights.

## License

[Your License Here]

