import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

class TransformerImputer(nn.Module):
    def __init__(self, input_dim, num_heads=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(dim_feedforward, input_dim)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        #print(f"X shape: {x.shape}")
        if len(x.shape) == 2:  # If input is 2D, add a batch dimension
            x = x.unsqueeze(0)  # Shape becomes [1, seq_len, embed_dim]
        
        if mask is not None:
            # Adjust `key_padding_mask` shape for unbatched case
            if mask.dim() == 2:  # Ensure it’s 1D if it’s unbatched data
                mask = mask.squeeze(0)
                try:
                    mask = mask.transpose(0, 1) # Transpose to (batch_size, sequence_length)
                    # select first row of mask
                    mask = mask[0]
                    mask = mask.unsqueeze(0)
                except:
                    mask = mask
                
            #print(f"Mask shape: {mask.shape}")
            #print(f"X shape: {x.shape}")
            x = self.transformer(x, src_key_padding_mask=mask)
            # remove batch dimension
            x = x.squeeze(0)
        else:
            x = self.transformer(x)
        return self.output(x)

class ImputationDataset(Dataset):
    def __init__(self, data, mask):
        self.data = torch.FloatTensor(data)
        self.mask = torch.BoolTensor(mask)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]

class DataImputer:
    def __init__(self, accuracy_threshold=0.7, batch_size=32, epochs=100):
        self.accuracy_threshold = accuracy_threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.models = {}
        
    def preprocess_data(self, df):
        processed_data = df.copy()
        mask = df.isna()
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in self.categorical_encoders:
                self.categorical_encoders[col] = LabelEncoder()
                non_null_data = df[col].dropna()
                self.categorical_encoders[col].fit(non_null_data)
            
            non_null_mask = ~df[col].isna()
            processed_data.loc[non_null_mask, col] = self.categorical_encoders[col].transform(
                df.loc[non_null_mask, col]
            )
            
        # Scale numerical variables
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            if col not in self.numerical_scalers:
                self.numerical_scalers[col] = StandardScaler()
                non_null_data = df[col].dropna().values.reshape(-1, 1)
                self.numerical_scalers[col].fit(non_null_data)
            
            # explicitly cast to a compatible dtype first
            processed_data[col] = processed_data[col].astype(float)
            non_null_mask = ~df[col].isna()
            processed_data.loc[non_null_mask, col] = self.numerical_scalers[col].transform(
                df.loc[non_null_mask, col].values.reshape(-1, 1)
            ).ravel()
            
        return processed_data, mask
    
    def train_and_evaluate(self, df, column):
        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare data
        non_null_rows = df[~df[column].isna()].index
        train_idx, val_idx = train_test_split(non_null_rows, test_size=0.2, random_state=42)
        
        # Create datasets
        train_data = ImputationDataset(
            df.loc[train_idx].values,
            df.loc[train_idx].isna().values
        )
        val_data = ImputationDataset(
            df.loc[val_idx].values,
            df.loc[val_idx].isna().values
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
    
        
        # Initialize model
        model = TransformerImputer(input_dim=df.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            model.train()
            for batch_data, batch_mask in train_loader:
                batch_data, batch_mask = batch_data.to(device), batch_mask.to(device)
                optimizer.zero_grad()
                output = model(batch_data, batch_mask)
                loss = criterion(output, batch_data)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_data, batch_mask in val_loader:
                    batch_data, batch_mask = batch_data.to(device), batch_mask.to(device)
                    output = model(batch_data, batch_mask)
                    val_loss = criterion(output, batch_data)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.models[column] = model
        
        # Evaluate accuracy
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_true = []
            for batch_data, batch_mask in val_loader:
                batch_data, batch_mask = batch_data.to(device), batch_mask.to(device)
                output = model(batch_data, batch_mask)
                output = output.cpu()
                batch_data = batch_data.cpu()
                val_preds.extend(output[:, df.columns.get_loc(column)].numpy())
                val_true.extend(batch_data[:, df.columns.get_loc(column)].numpy())
                print(f"Output shape: {output.shape}")
                print(f"Batch data shape: {batch_data.shape}")
            
            if column in self.categorical_encoders:
                val_preds = np.round(val_preds)
                accuracy = np.mean(np.array(val_preds) == np.array(val_true))
            else:
                accuracy = 1 - (np.mean(np.abs(np.array(val_preds) - np.array(val_true))))
        
        return accuracy
    
    def impute(self, df):
        processed_data, original_mask = self.preprocess_data(df)
        imputed_data = processed_data.copy()
        
        for column in df.columns:
            if df[column].isna().any():
                accuracy = self.train_and_evaluate(processed_data, column)
                
                if accuracy >= self.accuracy_threshold:
                    # Impute missing values
                    missing_rows = df[df[column].isna()].index
                    missing_data = ImputationDataset(
                        processed_data.loc[missing_rows].values,
                        processed_data.loc[missing_rows].isna().values
                    )
                    missing_loader = DataLoader(missing_data, batch_size=self.batch_size)
                    
                    model = self.models[column]
                    model.eval()
                    with torch.no_grad():
                        for batch_data, batch_mask in missing_loader:
                            output = model(batch_data, batch_mask)
                            imputed_values = output[:, processed_data.columns.get_loc(column)]
                            
                            if column in self.categorical_encoders:
                                imputed_values = np.round(imputed_values.numpy())
                                imputed_values = self.categorical_encoders[column].inverse_transform(imputed_values)
                            else:
                                imputed_values = self.numerical_scalers[column].inverse_transform(
                                    imputed_values.numpy().reshape(-1, 1)
                                ).ravel()
                            
                            imputed_data.loc[missing_rows, column] = imputed_values
                else:
                    print(f"Column {column} did not meet accuracy threshold ({accuracy:.2f} < {self.accuracy_threshold})")
                    # Restore original NaN values
                    imputed_data.loc[original_mask[column], column] = np.nan
                    
        return imputed_data

# Example usage
def impute_missing_values(df, accuracy_threshold=0.7, batch_size=32, epochs=100):
    """
    Impute missing values in a DataFrame using transformer-based models.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with missing values
    accuracy_threshold : float
        Minimum accuracy required to accept imputed values
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with imputed values where accuracy meets threshold
    """
    imputer = DataImputer(
        accuracy_threshold=accuracy_threshold,
        batch_size=batch_size,
        epochs=epochs
    )
    return imputer.impute(df)
