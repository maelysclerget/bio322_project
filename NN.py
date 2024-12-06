""" 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim 
import tqdm 
import torch
from torch.utils.data import Dataset, DataLoader
import optuna
from sklearn.model_selection import KFold
import copy
import warnings
from preprocessing import preprocessing_v1, apply_log_transformation

def neural_network():
    
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_correlation=True, apply_scaling=True, apply_remove_outliers=True, apply_variance_threshold=False, apply_random_forest=True)
    
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    y_train_std = StandardScaler().fit(y_train.values.reshape(-1, 1))
    
    # Standardize y_train
    y_train_nn = y_train_std.transform(y_train.values.reshape(-1, 1))

        
    X_train, y_train_nn, X_test = map(
        lambda array: torch.tensor(array, dtype=torch.float32),
        [X_train.values, y_train_nn, X_test.values]
    )  
    
    training_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train_nn),
        batch_size = 32,
        shuffle = True,
    )
    
    class NeuralNet1(torch.nn.Module):
        def __init__(self):
            super(NeuralNet1, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(X_train.shape[1], 64),
                torch.nn.Dropout(0.5),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.layers(x)
    
    epochs = 100
    
    model = NeuralNet1()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)
    
    model.train()
    for epoch in tqdm.tqdm(range(epochs)):
        for data, target in training_dataloader:
            optimizer.zero_grad()
            pred = model(data)
            loss = torch.sqrt(loss_func(pred, target))
            loss.backward()
            optimizer.step()
            
    # Evaluation
    with torch.no_grad():
        model.eval()
        train_preds = y_train_std.inverse_transform(model(X_train).detach().numpy())
        train_gt = y_train_std.inverse_transform(y_train_nn.detach().numpy())
        train_loss = np.sqrt(mean_squared_error(train_preds, train_gt))

        test_preds = y_train_std.inverse_transform(model(X_test).detach().numpy())

    print(f'{train_loss=}')
    submission = submission_file(test_preds.squeeze())
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/sample_submission_NN.csv', index=False)
    print('Submission file saved successfully.')
    
    def nn_early_stopping():

    # Preprocessing
    X_train, X_test, y_train = preprocessing_v1(
        apply_one_hot=True,
        apply_correlation=True,
        apply_scaling=True,
        apply_remove_outliers=False,
        apply_variance_threshold=False,
        apply_random_forest=True
    )
    
    # Drop sample_name column
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    # Standardize y_train
    y_train_std = StandardScaler().fit(y_train.values.reshape(-1, 1))
    y_train_nn = y_train_std.transform(y_train.values.reshape(-1, 1))
    
    # Convert to PyTorch tensors
    X_train, y_train_nn, X_test = map(
        lambda array: torch.tensor(array, dtype=torch.float32),
        [X_train.values, y_train_nn, X_test.values]
    )  
    
    # Dataloader
    training_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train_nn),
        batch_size=32,
        shuffle=True,
    )
    
    # Define the Neural Network
    class NeuralNet(nn.Module):
        def __init__(self, input_dim):
            super(NeuralNet, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Dropout(0.5),  # Dropout to reduce overfitting
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.Dropout(0.5),  # Dropout to reduce overfitting
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.Linear(32, 1)  # No activation on output layer
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Initialize the model, loss, and optimizer
    model = NeuralNet(X_train.shape[1])
    loss_func = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)
    
    # Early Stopping Parameters
    patience = 10
    min_delta = 1e-4
    best_loss = float('inf')
    early_stopping_counter = 0
    
    # Store training loss for plotting
    training_losses = []
    
    # Training Loop with Early Stopping
    epochs = 100
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in training_dataloader:
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_func(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Average loss for the epoch
        epoch_loss /= len(training_dataloader)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        # Check for early stopping
        if best_loss - epoch_loss > min_delta:
            best_loss = epoch_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered!")
                break
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Evaluation
    with torch.no_grad():
        model.eval()
        train_preds = y_train_std.inverse_transform(model(X_train).detach().numpy())
        train_gt = y_train_std.inverse_transform(y_train_nn.detach().numpy())
        train_loss = np.sqrt(mean_squared_error(train_preds, train_gt))

        test_preds = y_train_std.inverse_transform(model(X_test).detach().numpy())
    
    print(f'{train_loss=}')
    
    # Save submission file
    submission = submission_file(test_preds.squeeze())
    submission.to_csv('/Users/georgialex/Dropbox/ML_project/MLepfl/epfl-bio-322-2024/sample_submission_NN.csv', index=False)
    print('Submission file saved successfully.')
    
def loss(y_pred, y_hat):
    return torch.sqrt(torch.mean((y_pred - y_hat) ** 2))

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row_X = self.X.iloc[idx]
        row_y = self.y.iloc[idx]
        x = torch.tensor(row_X, dtype=torch.float)
        y = torch.tensor(row_y, dtype=torch.float)
        return x, y

# Neural Network
class Network(torch.nn.Module):
    def __init__(self, N_init, N1, N2, N3, N4):
        super(Network, self).__init__()
        self.linear1 = torch.nn.Linear(N_init, N1)
        self.linear2 = torch.nn.Linear(N1, N2)
        self.linear3 = torch.nn.Linear(N2, N3)
        self.linear4 = torch.nn.Linear(N3, N4)
        self.linear5 = torch.nn.Linear(N4, 1)  # Output layer

    def forward(self, x):
        z = torch.nn.functional.selu(self.linear1(x))
        z = torch.nn.functional.selu(self.linear2(z))
        z = torch.nn.functional.selu(self.linear3(z))
        z = torch.nn.functional.selu(self.linear4(z))
        z = self.linear5(z)
        return z

# Early Stopping Class
class EarlyStopTraining:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

def objective(trial, train_data, y_train):
    
    my_seed = 42
    num_epochs = 90
    batch_size = 32
    
    N1 = trial.suggest_int('N1', 80, 170)
    N2 = trial.suggest_int('N2', 50, 120)
    N3 = trial.suggest_int('N3', 20, 80)
    N4 = trial.suggest_int('N4', 5, 20)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=my_seed)
    val_losses = []

    for train_index, val_index in kf.split(train_data):
        # Split data into training and validation sets
        train_fold = train_data.iloc[train_index]
        val_fold = train_data.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]

        # Create Datasets and DataLoaders
        train_dataset = CustomDataset(train_fold, y_train_fold)
        val_dataset = CustomDataset(val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, optimizer, and early stopping
        model = Network(train_fold.shape[1], N1, N2, N3, N4)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
        early_stopping = EarlyStopTraining(patience=10)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss_avg = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss_value = loss(y_pred, y_batch.unsqueeze(-1))
                loss_value.backward()
                optimizer.step()
                train_loss_avg += loss_value.item()

            train_loss_avg /= len(train_loader)
            early_stopping(loss_value, model)

            if early_stopping.early_stop:
                break

        # Load the best model and evaluate on validation set
        model.load_state_dict(early_stopping.best_model)
        model.eval()

        x_val = torch.tensor(val_fold.values, dtype=torch.float)
        y_val_pred = model(x_val)
        val_loss = loss(y_val_pred, torch.tensor(y_val_fold.values).unsqueeze(-1))
        val_losses.append(val_loss.item())

    return np.mean(val_losses)

def optimisation():
    
    # Seed for reproducibility
    my_seed = 42
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    
     # Preprocess once before optimization
    train_data, _, y_train = preprocessing_v1(
        apply_one_hot=True,
        apply_scaling=True,
        apply_correlation=True,
        apply_random_forest=True
    )
    train_data = train_data.drop(columns=["sample_name"])

    # Wrap the objective function to pass additional arguments
    def wrapped_objective(trial):
        return objective(trial, train_data, y_train)

    # Run Optuna study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=my_seed))
    study.optimize(wrapped_objective, n_trials=20)

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    
def submission_nn_optimized():
    # Best parameters obtained from Optuna
    best_params = {
        'N1': 170,
        'N2': 101,
        'N3': 43,
        'N4': 16,
        'learning_rate': 0.000557849035390674,
    }
    
    # Preprocessing
    X_train, X_test, y_train = preprocessing_v1(
        apply_one_hot=True,
        apply_correlation=True,
        apply_scaling=True,
        apply_remove_outliers=False,
        apply_variance_threshold=False,
        apply_random_forest=True
    )
    
    # Drop unnecessary columns
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    # Standardize y_train
    y_train_std = StandardScaler().fit(y_train.values.reshape(-1, 1))
    y_train_nn = y_train_std.transform(y_train.values.reshape(-1, 1))

    # Convert data to PyTorch tensors
    X_train, y_train_nn, X_test = map(
        lambda array: torch.tensor(array, dtype=torch.float32),
        [X_train.values, y_train_nn, X_test.values]
    )  
    
    # DataLoader for training
    training_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train_nn),
        batch_size=32,
        shuffle=True,
    )
    
    # Define Neural Network with the best parameters
    class NeuralNetOptimized(torch.nn.Module):
        def __init__(self, input_dim, N1, N2, N3, N4):
            super(NeuralNetOptimized, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, N1),
                torch.nn.ReLU(),
                torch.nn.Linear(N1, N2),
                torch.nn.ReLU(),
                torch.nn.Linear(N2, N3),
                torch.nn.ReLU(),
                torch.nn.Linear(N3, N4),
                torch.nn.ReLU(),
                torch.nn.Linear(N4, 1),
            )
        def forward(self, x):
            return self.layers(x)
    
    # Initialize the model
    model = NeuralNetOptimized(
        input_dim=X_train.shape[1],
        N1=best_params['N1'],
        N2=best_params['N2'],
        N3=best_params['N3'],
        N4=best_params['N4'],
    )
    
    # Define loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=1e-3,
    )
    
    # Training loop
    epochs = 100
    model.train()
    for epoch in tqdm.tqdm(range(epochs), desc="Training Epochs"):
        for data, target in training_dataloader:
            optimizer.zero_grad()
            pred = model(data)
            loss = torch.sqrt(loss_func(pred, target))  # RMSE
            loss.backward()
            optimizer.step()
    
    # Evaluation
    with torch.no_grad():
        model.eval()
        # Predict on training set
        train_preds = y_train_std.inverse_transform(model(X_train).detach().numpy())
        train_gt = y_train_std.inverse_transform(y_train_nn.detach().numpy())
        train_loss = np.sqrt(mean_squared_error(train_preds, train_gt))

        # Predict on test set
        test_preds = y_train_std.inverse_transform(model(X_test).detach().numpy())

    # Print training loss
    print(f'{train_loss=}')
    
    # Save submission file
    submission = submission_file(test_preds.squeeze())
    submission.to_csv('/Users/georgialex/Dropbox/ML_project/MLepfl/epfl-bio-322-2024/sample_submission_NN.csv', index=False)
    print('Submission file saved successfully.')

    
    
      
def main():
        neural_network()
    
if __name__ == '__main__':
        main() 
         """