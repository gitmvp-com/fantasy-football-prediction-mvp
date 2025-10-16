import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class WeightedMSELoss(nn.Module):
    def __init__(self, high_threshold=15, low_threshold=4, high_weight=4, low_weight=1, very_low_weight=3):
        super(WeightedMSELoss, self).__init__()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.very_low_weight = very_low_weight
    
    def forward(self, outputs, targets):
        squared_diff = (outputs - targets) ** 2
        weights = torch.where(targets > self.high_threshold, self.high_weight, 
                              torch.where(targets < self.low_threshold, self.very_low_weight, self.low_weight))
        weighted_squared_diff = weights * squared_diff
        loss = torch.mean(weighted_squared_diff)
        return loss

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, learning_rate=0.001):
    """
    Train the LSTM model with early stopping.
    
    Args:
        model: LSTM model instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        Trained model
    """
    criterion = WeightedMSELoss(high_threshold=15, low_threshold=4.0, high_weight=4, low_weight=1, very_low_weight=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return model
