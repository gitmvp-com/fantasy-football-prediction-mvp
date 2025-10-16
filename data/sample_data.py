import pandas as pd
import numpy as np

def generate_sample_data(num_players=100, num_years=5):
    """
    Generate synthetic fantasy football data for testing the MVP.
    
    Args:
        num_players: Number of unique players to generate
        num_years: Number of years of historical data
    
    Returns:
        DataFrame with player statistics
    """
    np.random.seed(42)
    
    data = []
    player_names = [f"Player_{i}" for i in range(1, num_players + 1)]
    
    for player in player_names:
        # Generate consistent player quality (some players are better than others)
        player_quality = np.random.uniform(0.3, 1.0)
        base_age = np.random.randint(22, 28)
        
        for year_offset in range(num_years):
            year = 2020 + year_offset
            age = base_age + year_offset
            
            # Age affects performance
            age_factor = 1.0 if age < 28 else max(0.7, 1.0 - (age - 28) * 0.05)
            
            # Games played (some injury variance)
            games = np.random.choice([16, 17], p=[0.3, 0.7]) if np.random.random() > 0.15 else np.random.randint(8, 16)
            
            # Statistics based on player quality and age
            targets = int(player_quality * np.random.randint(80, 150) * age_factor)
            receptions = int(targets * np.random.uniform(0.6, 0.75))
            rec_yards = int(receptions * np.random.uniform(10, 15))
            rec_tds = int(player_quality * np.random.randint(3, 12) * age_factor)
            
            # Per game stats
            tgt_per_game = targets / games if games > 0 else 0
            rec_yds_per_game = rec_yards / games if games > 0 else 0
            td_per_game = rec_tds / games if games > 0 else 0
            
            # Fantasy points (Half PPR)
            fant_pt_half = (receptions * 0.5) + rec_yards * 0.1 + rec_tds * 6
            fant_pt_per_game = fant_pt_half / games if games > 0 else 0
            
            data.append({
                'Player': player,
                'Year': year,
                'Age': age,
                'G': games,
                'Tgt': targets,
                'Rec': receptions,
                'RecYds': rec_yards,
                'RecTD': rec_tds,
                'TD/G': round(td_per_game, 2),
                'RecYds/G': round(rec_yds_per_game, 1),
                'FantPtHalf/G': round(fant_pt_per_game, 1),
                'Tgt/G': round(tgt_per_game, 1)
            })
    
    df = pd.DataFrame(data)
    return df

def add_rolling_features(df):
    """
    Add rolling average features for better predictions.
    """
    df = df.sort_values(['Player', 'Year']).reset_index(drop=True)
    
    # Calculate rolling 2-year averages
    df['FantPtHalf/GLast2Y'] = df.groupby('Player')['FantPtHalf/G'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean().shift(1)
    )
    df['Tgt/GLast2Y'] = df.groupby('Player')['Tgt/G'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean().shift(1)
    )
    df['RecYds/GLast2Y'] = df.groupby('Player')['RecYds/G'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean().shift(1)
    )
    
    # Number of years played
    df['#ofY'] = df.groupby('Player').cumcount() + 1
    
    # Create target: next year's fantasy points per game
    df['NextYearFantPt/G'] = df.groupby('Player')['FantPtHalf/G'].shift(-1)
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df
