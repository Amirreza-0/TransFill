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
        self.input_dim = input_dim
        self.dim_feedforward = dim_feedforward

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
        if not isinstance(x, torch.FloatTensor):
            x = x.float()

        # Reshape input to (batch_size, sequence_length=1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Apply embedding
        batch_size, seq_len, features = x.shape
        x = self.embedding(x)  # Shape: (batch_size, seq_len, dim_feedforward)

        # Handle mask
        if mask is not None:
            # Reshape mask to (batch_size, sequence_length=1)
            if len(mask.shape) == 2:
                mask = mask.any(dim=-1).unsqueeze(1)
            elif len(mask.shape) == 3:
                mask = mask.any(dim=-1)

            # Ensure mask has correct shape
            if mask.shape != (batch_size, seq_len):
                mask = mask.new_zeros((batch_size, seq_len))

        # Apply transformer
        x = self.transformer(x)  # No need for src_key_padding_mask here since seq_len=1

        # Generate output
        x = self.output(x)

        # Remove sequence dimension
        x = x.squeeze(1)

        return x


class ImputationDataset(Dataset):
    def __init__(self, data, mask):
        self.data = torch.FloatTensor(np.array(data, dtype=np.float32))
        self.mask = torch.BoolTensor(mask.astype(bool))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]


class DataImputer:
    def __init__(self, accuracy_threshold=0.7, batch_size=32, epochs=100, device=None):
        self.accuracy_threshold = accuracy_threshold
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.models = {}

    def _encode_categorical(self, df, column):
        if column not in self.categorical_encoders:
            self.categorical_encoders[column] = LabelEncoder()
            non_null_data = df[column].dropna()
            self.categorical_encoders[column].fit(non_null_data)

        result = pd.Series(index=df.index, dtype=float)
        non_null_mask = ~df[column].isna()
        result[non_null_mask] = self.categorical_encoders[column].transform(
            df.loc[non_null_mask, column]
        )
        return result

    def _scale_numerical(self, df, column):
        if column not in self.numerical_scalers:
            self.numerical_scalers[column] = StandardScaler()
            non_null_data = df[column].dropna().values.reshape(-1, 1)
            self.numerical_scalers[column].fit(non_null_data)

        result = pd.Series(index=df.index, dtype=float)
        non_null_mask = ~df[column].isna()
        result[non_null_mask] = self.numerical_scalers[column].transform(
            df.loc[non_null_mask, column].values.reshape(-1, 1)
        ).ravel()
        return result

    def preprocess_data(self, df):
        processed_data = pd.DataFrame(index=df.index)
        mask = df.isna()

        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            processed_data[col] = self._encode_categorical(df, col)

        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            processed_data[col] = self._scale_numerical(df, col)

        return processed_data, mask

    def train_and_evaluate(self, df, column):
        non_null_rows = df[~df[column].isna()].index
        train_idx, val_idx = train_test_split(non_null_rows, test_size=0.2, random_state=42)

        train_data = ImputationDataset(
            df.loc[train_idx].values,
            df.loc[train_idx].isna().values
        )
        val_data = ImputationDataset(
            df.loc[val_idx].values,
            df.loc[val_idx].isna().values
        )

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)

        model = TransformerImputer(input_dim=df.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            model.train()
            for batch_data, batch_mask in train_loader:
                batch_data = batch_data.to(self.device)
                batch_mask = batch_mask.to(self.device)

                optimizer.zero_grad()
                output = model(batch_data, batch_mask)
                loss = criterion(output, batch_data)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_data, batch_mask in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_mask = batch_mask.to(self.device)

                    output = model(batch_data, batch_mask)
                    val_loss = criterion(output, batch_data)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.models[column] = model

        # Calculate accuracy
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_true = []

            for batch_data, batch_mask in val_loader:
                batch_data = batch_data.to(self.device)
                batch_mask = batch_mask.to(self.device)

                output = model(batch_data, batch_mask)
                col_idx = df.columns.get_loc(column)

                val_preds.extend(output[:, col_idx].cpu().numpy())
                val_true.extend(batch_data[:, col_idx].cpu().numpy())

            if column in self.categorical_encoders:
                val_preds = np.round(val_preds)
                accuracy = np.mean(np.array(val_preds) == np.array(val_true))
            else:
                accuracy = 1 - np.mean(np.abs(np.array(val_preds) - np.array(val_true)))

        return accuracy

    def impute(self, df):
        if df.empty:
            return df

        processed_data, original_mask = self.preprocess_data(df)
        imputed_data = processed_data.copy()

        for column in df.columns:
            if df[column].isna().any():
                accuracy = self.train_and_evaluate(processed_data, column)

                if accuracy >= self.accuracy_threshold:
                    missing_rows = df[df[column].isna()].index
                    if len(missing_rows) > 0:
                        missing_data = ImputationDataset(
                            processed_data.loc[missing_rows].values,
                            processed_data.loc[missing_rows].isna().values
                        )
                        missing_loader = DataLoader(missing_data, batch_size=self.batch_size)

                        model = self.models[column]
                        model.eval()

                        with torch.no_grad():
                            for batch_data, batch_mask in missing_loader:
                                batch_data = batch_data.to(self.device)
                                batch_mask = batch_mask.to(self.device)

                                output = model(batch_data, batch_mask)
                                imputed_values = output[:, processed_data.columns.get_loc(column)]

                                if column in self.categorical_encoders:
                                    imputed_values = np.round(imputed_values.cpu().numpy())
                                    imputed_values = self.categorical_encoders[column].inverse_transform(imputed_values)
                                else:
                                    imputed_values = self.numerical_scalers[column].inverse_transform(
                                        imputed_values.cpu().numpy().reshape(-1, 1)
                                    ).ravel()

                                imputed_data.loc[missing_rows, column] = imputed_values
                else:
                    print(
                        f"Column {column} did not meet accuracy threshold ({accuracy:.2f} < {self.accuracy_threshold})")
                    imputed_data.loc[original_mask[column], column] = np.nan

        return imputed_data


def impute_missing_values(df, accuracy_threshold=0.7, batch_size=32, epochs=100, device=None):
    imputer = DataImputer(
        accuracy_threshold=accuracy_threshold,
        batch_size=batch_size,
        epochs=epochs,
        device=device
    )
    return imputer.impute(df)


# Example usage
df = pd.DataFrame({
    'age': [25, np.nan, 35],
    'income': [50000, 60000, np.nan],
    'department': ['Sales', np.nan, 'Engineering']
})

imputed_df = impute_missing_values(
    df,
    accuracy_threshold=0.7,
    batch_size=32,
    epochs=100
)
