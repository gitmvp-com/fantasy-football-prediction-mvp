# Fantasy Football Prediction MVP

A simplified MVP version of a fantasy football prediction system using LSTM neural networks to predict next year's fantasy points per game for wide receivers.

## Features

- **LSTM Neural Network**: Predicts fantasy football points using historical player statistics
- **Sample Data**: Includes synthetic data generator for immediate testing
- **Training & Evaluation**: Train model and evaluate predictions
- **Simple Predictions**: Generate predictions for upcoming season

## Project Structure

```
├── data/
│   └── sample_data.py       # Generates synthetic training data
├── scripts/
│   ├── model.py            # LSTM model architecture
│   ├── training.py         # Model training logic
│   └── prediction.py       # Prediction functions
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gitmvp-com/fantasy-football-prediction-mvp.git
cd fantasy-football-prediction-mvp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train the model and generate predictions:

```bash
python main.py
```

This will:
1. Generate synthetic training data
2. Train an LSTM model
3. Evaluate the model on test data
4. Generate predictions for the next season
5. Display top predicted players

## Requirements

- Python 3.x
- torch==2.3.1
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.5.0

## How It Works

1. **Data Preparation**: Sample player statistics are generated including yards, touchdowns, targets, etc.
2. **Feature Engineering**: Rolling averages and historical stats are calculated
3. **LSTM Model**: A multi-layer LSTM network learns patterns from historical data
4. **Prediction**: The trained model predicts fantasy points for the next season

## Model Architecture

- **Input Layer**: Player statistics (games, targets, receptions, yards, TDs, etc.)
- **LSTM Layers**: 2 layers with 128 hidden units and dropout
- **Fully Connected Layers**: 4 dense layers (128 → 64 → 32 → 1)
- **Output**: Predicted fantasy points per game

## Future Enhancements

This is an MVP version. The full version could include:
- Real NFL player data integration
- Advanced feature engineering
- Multiple position support (RB, TE, QB)
- Web interface for predictions
- Historical accuracy tracking
- Player comparison tools

## License

MIT License
