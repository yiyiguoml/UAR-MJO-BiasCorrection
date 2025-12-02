"""
MJO Ensemble Bias Correction using Attention-based RNN

This script implements a multi-head attention RNN model for correcting bias in
Madden-Julian Oscillation (MJO) ensemble forecasts. The model processes ensemble
member data and learns to predict improved MJO indices.

References:
- Madden-Julian Oscillation (MJO): A dominant mode of intraseasonal variability
  in the tropical atmosphere
- Bivariate Mean Squared Error (BMSE): Error metric for phase-amplitude data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pathlib
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Seed for reproducibility
seed = 1102

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for sequence processing.

    This module implements scaled dot-product attention with multiple heads,
    allowing the model to attend to different aspects of the input sequence.

    Args:
        hidden_size (int): Size of hidden layer
        num_heads (int): Number of attention heads
        bidirectional (bool): Whether RNN is bidirectional (default: False)
    """
    def __init__(self, hidden_size, num_heads, bidirectional=False):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear layers for query, key, and value projections
        self.query = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.key = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.value = nn.Linear(hidden_size * (2 if bidirectional else 1), hidden_size)

        # Output linear layer to combine multi-head outputs
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, rnn_out):
        """
        Forward pass of multi-head attention.

        Args:
            rnn_out (torch.Tensor): RNN output of shape (batch_size, seq_len, hidden_size)

        Returns:
            pooled_context (torch.Tensor): Attended context pooled across sequence (batch_size, hidden_size)
            attn_weights (torch.Tensor): Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        
        batch_size, seq_len, hidden_size = rnn_out.size()
        
        # Compute queries, keys, and values
        queries = self.query(rnn_out)  # (batch_size, seq_len, hidden_size)
        keys = self.key(rnn_out)      # (batch_size, seq_len, hidden_size)
        values = self.value(rnn_out)  # (batch_size, seq_len, hidden_size)
        
        # Reshape for multi-head attention: (batch_size, seq_len, num_heads, head_dim)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        scores = scores / (self.head_dim ** 0.5)  # Scale by sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)  # Normalize over seq_len
        
        # Weighted sum of values
        context = torch.matmul(attn_weights, values)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate multi-head outputs
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Final linear layer
        context = self.out(context)  # Combine multi-head outputs
        
        # Pool context across the sequence
        pooled_context = torch.sum(context, dim=1)  # (batch_size, hidden_size)
        
        return pooled_context, attn_weights

class SharedAttentionRNN(nn.Module):
    """
    RNN with multi-head attention for MJO bias correction.

    This model processes ensemble member sequences through an RNN, applies
    multi-head attention to capture important temporal features, and outputs
    bias-corrected MJO predictions.

    Args:
        input_size (int): Number of input features
        hidden_size (int): Size of RNN hidden state
        output_size (int): Number of output features (typically 2 for MJO indices)
        num_layers (int): Number of RNN layers (default: 2)
        nonlinearity (str): RNN nonlinearity ('tanh' or 'relu', default: 'tanh')
        bidirectional (bool): Whether to use bidirectional RNN (default: False)
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, nonlinearity='tanh', bidirectional=False):
        super(SharedAttentionRNN, self).__init__()

        # RNN layer
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(hidden_size, 2, bidirectional)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        """
        Forward pass through RNN and attention layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            output (torch.Tensor): Predicted MJO indices (batch_size, output_size)
        """
        # RNN forward pass
        rnn_out, _ = self.rnn(x)

        # Attention forward pass
        context, attn_weights = self.attention(rnn_out)

        # Output prediction
        output = self.fc(context)

        return output
    
# Model hyperparameters
input_seq = 34
input_dim = 4
output_dim = 2
hidden_size = 40
total_epoch = 2000
lr = 0.001
la = 0.5

# Loss and Evaluation Functions

def tor_bmsea(x1, x2, y1, y2):
    """
    Bivariate Mean Squared Error - Amplitude component (PyTorch version).

    Computes the MSE between predicted and observed amplitudes.

    Args:
        x1, x2 (torch.Tensor): Predicted MJO components (real, imaginary)
        y1, y2 (torch.Tensor): Observed MJO components (real, imaginary)

    Returns:
        error (torch.Tensor): BMSE amplitude error
    """
    Am = torch.sqrt(x1**2 + x2**2)
    Ao = torch.sqrt(y1**2 + y2**2)
    mse = (Am - Ao)**2
    error = torch.mean(mse)
    return error

def bmsea(x1, x2, y1, y2):
    """
    Bivariate Mean Squared Error - Amplitude component (NumPy version).

    Args:
        x1, x2 (np.ndarray): Predicted MJO components (real, imaginary)
        y1, y2 (np.ndarray): Observed MJO components (real, imaginary)

    Returns:
        error (float): BMSE amplitude error
    """
    Am = np.sqrt(x1**2 + x2**2)
    Ao = np.sqrt(y1**2 + y2**2)
    error = np.mean((Am - Ao)**2)
    return error

def tor_bmse(x1, x2, y1, y2):
    """
    Complete Bivariate Mean Squared Error (amplitude + phase).

    Args:
        x1, x2 (torch.Tensor): Predicted MJO components
        y1, y2 (torch.Tensor): Observed MJO components

    Returns:
        bmse (torch.Tensor): Total BMSE error
    """
    amp_pred = torch.sqrt(x1**2 + x2**2)
    amp_obs = torch.sqrt(y1**2 + y2**2)
    phase_diff = torch.atan2(x2, x1) - torch.atan2(y2, y1)

    bmse = torch.mean((amp_pred - amp_obs)**2) + \
           torch.mean(2 * amp_pred * amp_obs * (1 - torch.cos(phase_diff)))
    return bmse

def tor_bmseb(x1, x2, y1, y2, lam):
    """
    Combined MSE and cosine similarity loss for training.

    Combines Euclidean distance between components with a cosine similarity term
    to capture both magnitude and directional errors.

    Args:
        x1, x2 (torch.Tensor): Predicted MJO components
        y1, y2 (torch.Tensor): Observed MJO components
        lam (float): Weight for cosine similarity term

    Returns:
        error (torch.Tensor): Total loss
        error_mse (torch.Tensor): MSE component
        error_cos (torch.Tensor): Cosine component
    """
    p = 2
    A1 = torch.abs(x1 - y1) ** p
    A2 = torch.abs(x2 - y2) ** p

    error_mse = torch.mean(A1 + A2)

    # Cosine similarity term
    norm_pred = torch.sqrt(x1**2 + x2**2)
    norm_obs = torch.sqrt(y1**2 + y2**2)
    error_cos = torch.mean(1 - ((x1 * y1 + x2 * y2) / (norm_pred * norm_obs)))

    error = error_mse + lam * error_cos
    return error, error_mse, error_cos

def bmseb(x1, x2, y1, y2):
    """
    Bivariate Mean Squared Error - Phase component (NumPy version).

    Args:
        x1, x2 (np.ndarray): Predicted MJO components
        y1, y2 (np.ndarray): Observed MJO components

    Returns:
        error (float): BMSE phase error
    """
    Am = np.sqrt(x1**2 + x2**2)
    Ao = np.sqrt(y1**2 + y2**2)
    thetam = np.arctan2(x2, x1)
    thetao = np.arctan2(y2, y1)
    error = np.mean(2 * Am * Ao * (1 - np.cos(thetam - thetao)))
    return error

def amp(x1, x2):
    """Calculate amplitude from two components."""
    Am = np.sqrt(x1**2 + x2**2)
    return Am

def rmse(x1, x2, y1, y2):
    """
    Root Mean Squared Error between predicted and observed MJO components.

    Args:
        x1, x2 (np.ndarray): Predicted MJO components
        y1, y2 (np.ndarray): Observed MJO components

    Returns:
        error (float): RMSE
    """
    error = np.sqrt(np.mean((y1 - x1)**2 + (y2 - x2)**2))
    return error 

# Training configuration
years = 15  # Number of test years (1996-2010)
num_runs = 5  # Number of independent runs for statistical robustness

# Initialize error tracking arrays
s2s_bmsea_error = np.empty((years))
bc_bmsea_error = np.empty((num_runs, years))
s2s_bmseb_error = np.empty((years))
bc_bmseb_error = np.empty((num_runs, years))
num = np.zeros((years))
rmse_loss = np.zeros([num_runs, 30])

# Accumulated errors across runs
sbmsea = 0
sbmseb = 0
bbmsea = 0
bbmseb = 0

# Loss tracking
testing_loss = np.zeros([num_runs, years, total_epoch])
training_loss = np.zeros([num_runs, years, total_epoch])
valid_errors = np.zeros([num_runs, years, total_epoch])


# Main training loop across multiple runs
for t in range(num_runs):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Run {t+1}/{num_runs}", flush=True)
    

    # Loop over test years
    for j in range(years):
        # Calculate target year (1996-2010)
        target_year = str(1981 + j + 30 - years)

        # Feature indices: [rmm1, rmm2, amplitude, lead_time]
        train_idx = [0, 1, 6, 9]

        # Load and reshape training data
        x_train = np.load(f'../preprocessing_data/target_y{target_year}.train.npy').astype(np.float32)
        new_shape = (x_train.shape[0] * x_train.shape[1], 34, 10)
        x_train = x_train.reshape(new_shape)
        # Extract training features and targets
        total_x = x_train[:, :, train_idx].reshape(-1, input_seq, input_dim)
        total_y = np.zeros((x_train.shape[0], 2))

        # Extract target values (observed MJO at longest available lead)
        for i in range(x_train.shape[0]):
            for z in range(33):
                if total_x[i, z+1, 3] != 0:  # Check if lead time exists
                    total_y[i, :] = x_train[i, z+1, 2:4]  # Extract observed values
                else:
                    break

        # Load and prepare validation data
        validation_data = np.load(f'../preprocessing_data/target_y{target_year}.validate.npy').astype(np.float32)
        new_shape = (validation_data.shape[0] * validation_data.shape[1], 34, 10)
        validation_data = validation_data.reshape(new_shape)
        valid_x = validation_data[:, :, train_idx].reshape(-1, input_seq, input_dim)
        valid_y = np.zeros((validation_data.shape[0], 2))

        for i in range(validation_data.shape[0]):
            for z in range(33):
                if validation_data[i, z+1, 3] != 0:
                    valid_y[i, :] = validation_data[i, z+1, 2:4]
                else:
                    break 

        # Load and prepare test data
        testing_data = np.load(f'../preprocessing_data/target_y{target_year}.test.npy').astype(np.float32)
        target_x = testing_data[:, :, train_idx].reshape(-1, input_seq, input_dim)

        # Extract S2S ensemble mean predictions for comparison
        test_xx = np.zeros((target_x.shape[0], 1, 2))
        for i in range(target_x.shape[0]):
            for z in range(33):
                if target_x[i, z+1, 3] != 0:
                    test_xx[i, 0, :] = target_x[i, z+1, 0:2]  # S2S prediction
                else:
                    break

        # Extract test targets
        target_y = np.zeros((testing_data.shape[0], 2))
        for i in range(testing_data.shape[0]):
            for z in range(33):
                if testing_data[i, z+1, 3] != 0:
                    target_y[i, :] = testing_data[i, z+1, 2:4]
                else:
                    break

        if target_x.size == 0:
            continue

        # Convert to PyTorch tensors
        train_x = torch.from_numpy(total_x).to(device)
        train_y = torch.from_numpy(total_y).to(device)
        test_x = torch.from_numpy(target_x).to(device)
        test_y = target_y
        valid_x = torch.from_numpy(valid_x).to(device)
        valid_y = torch.from_numpy(valid_y).to(device)
        test_s2sx = torch.from_numpy(test_xx).to(device)

        # Hyperparameter search over lambda (cosine loss weight)
        lam = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        best_bmse = 100
        model_states = []
        autonum = 0 
        # Train model for each lambda value
        for lam_ind in range(6):
            best_error = 100
            temp_states = []
            model = SharedAttentionRNN(input_dim, hidden_size, output_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.9))

            counter = 0
            epo = 30  # Start saving models after epoch 30
            patience = 20  # Early stopping patience

            for epoch in range(total_epoch):
                model.train()
                optimizer.zero_grad()

                # Forward pass
                outputs = model(train_x)
                loss, loss_mse, loss_cos = tor_bmseb(
                    outputs[:, 0], outputs[:, 1],
                    train_y[:, 0], train_y[:, 1],
                    lam[lam_ind]
                )

                # Validation
                valid_outputs = model(valid_x)
                valid_loss, valid_mse, valid_cos = tor_bmseb(
                    valid_outputs[:, 0], valid_outputs[:, 1],
                    valid_y[:, 0], valid_y[:, 1],
                    lam[lam_ind]
                )

                # Track losses
                training_loss[t, j, epoch] = loss.cpu().detach().numpy()
                valid_errors[t, j, epoch] = valid_loss.cpu().detach().numpy()

                # Compute BMSE for model selection
                validation_outputs = tor_bmse(
                    valid_outputs[:, 0], valid_outputs[:, 1],
                    valid_y[:, 0], valid_y[:, 1]
                )

                # Save model checkpoints and check for early stopping
                if epoch >= epo:
                    temp_states.append(model.state_dict())
                    modelindx = epoch - epo
                    valid_lam = valid_errors[t, j, epoch]

                    if valid_lam <= best_error:
                        best_error = valid_lam
                        tempautonum = modelindx
                        counter = 0
                        validation_bmse = validation_outputs
                    else:
                        counter += 1

                    if counter >= patience:
                        # Select best model across lambda values
                        if validation_bmse <= best_bmse:
                            best_bmse = validation_bmse
                            model_states = temp_states
                            autonum = tempautonum
                        break

                # Backward pass
                loss.backward()
                optimizer.step()

        # Save best model
        output_dir = pathlib.Path(f'../outputs/Ours_ic_Nosquare/{t+1}')
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f'{target_year}.pth'
        torch.save(model_states[autonum], save_path)

        # Evaluate on test set
        model.load_state_dict(model_states[autonum])
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_x)

            test_s2sx = test_s2sx.cpu().numpy()
            test_outputs = test_outputs.cpu().numpy().reshape(-1, 2)

            num[j] = test_x.shape[0]

            # Compute errors for bias-corrected and S2S predictions
            bc_bmsea_error[t, j] = bmsea(test_outputs[:, 0], test_outputs[:, 1],
                                         test_y[:, 0], test_y[:, 1]) * test_x.shape[0]
            s2s_bmsea_error[j] = bmsea(test_s2sx[:, 0, 0], test_s2sx[:, 0, 1],
                                       test_y[:, 0], test_y[:, 1]) * test_x.shape[0]

            bc_bmseb_error[t, j] = bmseb(test_outputs[:, 0], test_outputs[:, 1],
                                         test_y[:, 0], test_y[:, 1]) * test_x.shape[0]
            s2s_bmseb_error[j] = bmseb(test_s2sx[:, 0, 0], test_s2sx[:, 0, 1],
                                       test_y[:, 0], test_y[:, 1]) * test_x.shape[0]

            rmse_loss[t, j] = rmse(test_outputs[:, 0], test_outputs[:, 1],
                                   test_y[:, 0], test_y[:, 1])

    # Compute run statistics
    s2s_BMSEa_lead = np.sum(s2s_bmsea_error) / np.sum(num)
    s2s_BMSEb_lead = np.sum(s2s_bmseb_error) / np.sum(num)
    bc_BMSEa_lead = np.sum(bc_bmsea_error[t]) / np.sum(num)
    bc_BMSEb_lead = np.sum(bc_bmseb_error[t]) / np.sum(num)

    # Accumulate for averaging across runs
    sbmsea += s2s_BMSEa_lead
    sbmseb += s2s_BMSEb_lead
    bbmsea += bc_BMSEa_lead
    bbmseb += bc_BMSEb_lead

    # Print run results
    print(f'Run {t+1} Results:', flush=True)
    print(f'  S2S BMSEa: {s2s_BMSEa_lead:.4f}', flush=True)
    print(f'  BC BMSEa:  {bc_BMSEa_lead:.4f}', flush=True)
    print(f'  S2S BMSEb: {s2s_BMSEb_lead:.4f}', flush=True)
    print(f'  BC BMSEb:  {bc_BMSEb_lead:.4f}', flush=True)
    print(f'  Total BC BMSE: {bc_BMSEa_lead + bc_BMSEb_lead:.4f}', flush=True)

    seed += 1

# Print final averaged results across all runs
print('Final Results (averaged over 5 runs):', flush=True)
print('='*50, flush=True)
print(f'S2S BMSEa: {sbmsea/num_runs:.4f}', flush=True)
print(f'BC BMSEa:  {bbmsea/num_runs:.4f}', flush=True)
print(f'S2S BMSEb: {sbmseb/num_runs:.4f}', flush=True)
print(f'BC BMSEb:  {bbmseb/num_runs:.4f}', flush=True)
print(f'Total S2S BMSE: {(sbmsea+sbmseb)/num_runs:.4f}', flush=True)
print(f'Total BC BMSE:  {(bbmsea+bbmseb)/num_runs:.4f}', flush=True)
print('='*50, flush=True)
