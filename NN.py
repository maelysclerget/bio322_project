
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import tqdm 
from preprocessing import preprocessing_v1, submission_file

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
    
              
def main():
        neural_network()
    
if __name__ == '__main__':
        main() 
        