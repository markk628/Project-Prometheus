import gc
import joblib
import math
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from src.config.config import ROOT_DIR, DATA_DIR, DEVICE
from src.utils.logger import Logger
from src.utils.utils import save_to_parquet, create_directory


# def nearest_power_of_2(x, min_val=16, max_val=1024):
#     power = 2 ** round(math.log2(x))
#     return int(min(max(power, min_val), max_val))

class _Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        h1, h2, h3, h4 = 2048, 1024, 512, 256

        self.l1  = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.l2  = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.l3  = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.l4  = nn.Linear(h3, h4)
        self.bn4 = nn.BatchNorm1d(h4)
        self.out = nn.Linear(h4, latent_dim)

        # Skip: project h1 → h3, bridges the two biggest compression steps
        self.skip = nn.Linear(h1, h3)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.05)

    def forward(self, x):
        x = self.drop(self.act(self.bn1(self.l1(x))))           # → 2048
        skip = x
        x = self.drop(self.act(self.bn2(self.l2(x))))           # → 1024
        x = self.act(self.bn3(self.l3(x))) + self.skip(skip)    # → 512 + skip
        x = self.act(self.bn4(self.l4(x)))                      # → 256
        return torch.tanh(self.out(x))                          # → latent ∈ [-1, 1]


class _Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        h1, h2, h3, h4 = 2048, 1024, 512, 256

        self.l1  = nn.Linear(latent_dim, h4)
        self.bn1 = nn.BatchNorm1d(h4)
        self.l2  = nn.Linear(h4, h3)
        self.bn2 = nn.BatchNorm1d(h3)
        self.l3  = nn.Linear(h3, h2)
        self.bn3 = nn.BatchNorm1d(h2)
        self.l4  = nn.Linear(h2, h1)
        self.bn4 = nn.BatchNorm1d(h1)
        self.out = nn.Linear(h1, input_dim)

        # Mirror skip: project h4 → h2
        self.skip = nn.Linear(h4, h2)
        self.act  = nn.GELU()

    def forward(self, z):
        x = self.act(self.bn1(self.l1(z)))                      # → 256
        skip = x
        x = self.act(self.bn2(self.l2(x)))                      # → 512
        x = self.act(self.bn3(self.l3(x))) + self.skip(skip)    # → 1024 + skip
        x = self.act(self.bn4(self.l4(x)))                      # → 2048
        return self.out(x)                                      # → input_dim


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, logger=None):
        super().__init__()

        if logger:
            logger.info(
                f"AE Architecture: {input_dim} → 2048 → 1024 → 512 → 256 → {latent_dim}"
            )

        self.encoder = _Encoder(input_dim, latent_dim)
        self.decoder = _Decoder(input_dim, latent_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class DataDimensionalReducer:
    def __init__(self, logger: Logger):
        self.data_dir = f"{DATA_DIR}/preprocessed/v4/"
        self.train_path = f"{self.data_dir}/unified/unified_train.parquet"
        self.valid_path = f"{self.data_dir}/unified/unified_valid.parquet"
        self.test_path = f"{self.data_dir}/unified/unified_test.parquet"
        self.logger = logger
        
        self.closes_dfs = None
        self.temporal_dfs = None
        
    def _fetch_unified_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self.logger.info("Fetching data...")
        train = pl.read_parquet(self.train_path)
        valid = pl.read_parquet(self.valid_path)
        test  = pl.read_parquet(self.test_path)
        self.logger.info("Data fetched.")
        return train, valid, test
    
    def _drop_unnecessary_features(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self.logger.info("Dropping unnecessary features...")
        
        closes_features = [feature for feature in train.columns if feature.endswith('_close') and feature != "minutes_to_close"]
        temporal_features = [
            "timestamp", 
            "minute_sin", "minute_cos",
            "hour_sin", "hour_cos",
            "day_sin", "day_cos",
            "minutes_since_open", "minutes_to_close"
        ]
        features_to_drop = temporal_features + closes_features
        
        train_closes = train.select(pl.col(closes_features))
        valid_closes = valid.select(pl.col(closes_features))
        test_closes = test.select(pl.col(closes_features))
        
        train_temporal = train.select(pl.col(temporal_features))
        valid_temporal = valid.select(pl.col(temporal_features))
        test_temporal = test.select(pl.col(temporal_features))
        
        self.closes_dfs = {
            'train': train_closes,
            'valid': valid_closes,
            'test': test_closes
        }
        
        self.temporal_dfs = {
            'train': train_temporal,
            'valid': valid_temporal,
            'test': test_temporal
        }
        
        self.logger.info(f"Dropped features: {features_to_drop}")
        return train.drop(features_to_drop), valid.drop(features_to_drop), test.drop(features_to_drop)

    def _prepare_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        self.logger.info('Preparing data...')
        train, valid, test = self._fetch_unified_data()

        self.logger.info(f"Feature count before reduction: {train.shape[1]}")
        return self._drop_unnecessary_features(train, valid, test)
    
    def _pca(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, version: int, incremental: bool=False) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

        self.logger.info("Running PCA...")

        train_np = train.to_numpy().astype(np.float32)
        valid_np = valid.to_numpy().astype(np.float32)
        test_np  = test.to_numpy().astype(np.float32)

        if incremental:
            pca = IncrementalPCA(
                n_components=150,
                batch_size=10000,
                whiten=False
            )
        else:
            pca = PCA(
                n_components=0.95,
                svd_solver="full",
                whiten=False
            )
        
        train_pca = pca.fit_transform(train_np)
        valid_pca = pca.transform(valid_np)
        test_pca  = pca.transform(test_np)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        self.logger.info(f"Top 10 explained variance ratios: {pca.explained_variance_ratio_[:10]}")
        self.logger.info(f"Last 10 explained variance ratios: {pca.explained_variance_ratio_[-10:]}")
        self.logger.info(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
        self.logger.info(f"PCA components for 95% variance: {n_95}")

        # Save PCA model
        joblib.dump(pca, f"{ROOT_DIR}/src/v4/preprocessing/dimensional_reduction/pca_v{version}.pkl")

        # Create column names
        columns = [f"pca_{i}" for i in range(train_pca.shape[1])]

        # Convert back to Polars
        train_pl = pl.DataFrame(train_pca, schema=columns)
        valid_pl = pl.DataFrame(valid_pca, schema=columns)
        test_pl  = pl.DataFrame(test_pca,  schema=columns)

        return train_pl, valid_pl, test_pl

    def _encode_in_batches(self, model, data_np, batch_size, device):
        self.logger.info("Encoding in batches...")
        
        model.eval()
        latents = []

        with torch.no_grad():
            for i in range(0, len(data_np), batch_size):
                batch = torch.from_numpy(data_np[i:i+batch_size]).to(device)
                z = model.encoder(batch)
                latents.append(z.cpu().numpy())

        return np.vstack(latents)
       
    def _autoencoder(
        self,
        train: pl.DataFrame,
        valid: pl.DataFrame,
        test: pl.DataFrame,
        latent_dim: int,
        version: int,
        epochs: int=20,
        batch_size: int=1024,
        lr: float=1e-3,
        encode_in_batches: bool=False
    ):

        self.logger.info("Training Autoencoder...")

        device = torch.device(DEVICE)
        self.logger.info(f"Using device: {device}")

        # Convert to numpy and free the DataFrames immediately (#8)
        full_train_np = train.to_numpy().astype(np.float32)
        valid_np = valid.to_numpy().astype(np.float32)
        test_np  = test.to_numpy().astype(np.float32)
        del train, valid, test
        gc.collect()

        # CHRONOLOGICAL SPLIT to prevent future-peeking leakage
        # Take the last 15% of the train set as the AE validation set
        split_idx = int(len(full_train_np) * 0.85)
        ae_train_np = full_train_np[:split_idx]
        ae_valid_np = full_train_np[split_idx:]

        model = Autoencoder(full_train_np.shape[1], latent_dim, self.logger).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        criterion = nn.MSELoss()

        # DataLoaders strictly using the AE's internal train/valid split
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(ae_train_np)),
            batch_size=batch_size,
            shuffle=True
        )
        
        valid_loader = DataLoader(
            TensorDataset(torch.from_numpy(ae_valid_np)),
            batch_size=batch_size,
            shuffle=False
        )
        
        save_path = f"{ROOT_DIR}/src/v4/preprocessing/dimensional_reduction/autoencoder_v{version}.pt"
        best_valid_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # ---- TRAIN ----
            model.train()
            train_loss = 0.0
            noise_std = 0.1 * (1 - epoch / epochs) + 0.01

            for (batch,) in train_loader:
                batch = batch.to(device)

                noise = torch.randn_like(batch) * noise_std
                noisy_batch = batch + noise

                optimizer.zero_grad()
                reconstructed = model(noisy_batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # ---- VALIDATION ----
            model.eval()
            valid_loss = 0.0

            with torch.no_grad():
                for (batch,) in valid_loader:
                    batch = batch.to(device)
                    reconstructed = model(batch)
                    loss = criterion(reconstructed, batch)
                    valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)  # (#7) full model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

            # Step scheduler on validation loss
            scheduler.step(valid_loss)
            current_lr = optimizer.param_groups[0]['lr']

            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | LR: {current_lr} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
            )
            
        # (#7) Reload best full model so recon loss uses matched encoder+decoder
        model.load_state_dict(torch.load(save_path, weights_only=True))
        model.eval()

        def _recon_loss(model, data_np, criterion, device, n=5000):
            idx = np.random.choice(len(data_np), size=min(n, len(data_np)), replace=False)
            sample = torch.from_numpy(data_np[idx]).to(device)
            recon = model(sample)
            return criterion(recon, sample).item()

        with torch.no_grad():
            train_recon = _recon_loss(model, full_train_np, criterion, device)
            valid_recon = _recon_loss(model, valid_np, criterion, device)
            test_recon  = _recon_loss(model, test_np,  criterion, device)

        self.logger.info(f"Reconstruction MSE | Train: {train_recon:.6f} | Valid: {valid_recon:.6f} | Test: {test_recon:.6f}")
        
        # Encode datasets
        self.logger.info("Encoding datasets...")
        
        if encode_in_batches:
            train_latent = self._encode_in_batches(model, full_train_np, batch_size, device)
            valid_latent = self._encode_in_batches(model, valid_np, batch_size, device)
            test_latent  = self._encode_in_batches(model, test_np,  batch_size, device)
        else:
            with torch.no_grad():
                train_latent = model.encoder(torch.from_numpy(full_train_np).to(device)).cpu().numpy()
                valid_latent = model.encoder(torch.from_numpy(valid_np).to(device)).cpu().numpy()
                test_latent  = model.encoder(torch.from_numpy(test_np).to(device)).cpu().numpy()
                
        latent_var = np.var(train_latent, axis=0)
        self.logger.info(f"Latent feature variance: {latent_var}")
        self.logger.info(f"Mean latent var: {latent_var.mean()}")
        self.logger.info(f"Min latent var: {latent_var.min()}")
        
        self.logger.info("Autoencoder training complete.")

        columns = [f"latent_{i}" for i in range(latent_dim)]

        train_pl = pl.DataFrame(train_latent, schema=columns)
        valid_pl = pl.DataFrame(valid_latent, schema=columns)
        test_pl  = pl.DataFrame(test_latent,  schema=columns)

        return train_pl, valid_pl, test_pl
    
    def _reduce_dimension_v1(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, latent_dim: int=32, epochs: int=20, batch_size: int=1024, lr: float=1e-3) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """raw features -> pca -> pca result -> autoencoder -> ae result"""
        # PCA
        train, valid, test = self._pca(train, valid, test, version=1)
        self.logger.info(f"Feature count raw features -> PCA: {train.shape[1]}")

        # Autoencoder
        train, valid, test = self._autoencoder(
            train,
            valid,
            test,
            latent_dim=latent_dim,
            version=1,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

        self.logger.info(f"Feature count pca features -> Autoencoder: {train.shape[1]}")
        
        scaler = StandardScaler()
        train_np = scaler.fit_transform(train.to_numpy())
        valid_np = scaler.transform(valid.to_numpy())
        test_np  = scaler.transform(test.to_numpy())

        train = pl.DataFrame(train_np, schema=train.columns)
        valid = pl.DataFrame(valid_np, schema=valid.columns)
        test  = pl.DataFrame(test_np,  schema=test.columns)

        joblib.dump(scaler, f"{ROOT_DIR}/src/v4/preprocessing/dimensional_reduction/scaler_v1.pkl")

        return train, valid, test

    def _reduce_dimension_v2(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, latent_dim: int=32, epochs: int=20, batch_size: int=1024, lr: float=1e-3) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        raw features -> pca -> pca result
        raw features -> autoencoder -> ae result
        hstack(pca result, ae result)
        """
        # PCA (makes its own numpy copies internally)
        train_pca, valid_pca, test_pca = self._pca(train, valid, test, version=2)
        self.logger.info(f"Feature count raw features -> PCA: {train_pca.shape[1]}")

        # (#8) AE will del the DataFrames after converting to numpy
        train_latent, valid_latent, test_latent = self._autoencoder(
            train,
            valid,
            test,
            latent_dim=latent_dim,
            version=2,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )
        # train, valid, test DataFrames are now freed inside _autoencoder

        # Concatenate PCA + AE features
        train = train_pca.hstack(train_latent)
        valid = valid_pca.hstack(valid_latent)
        test  = test_pca.hstack(test_latent)
        del train_pca, valid_pca, test_pca, train_latent, valid_latent, test_latent
        gc.collect()

        self.logger.info(f"Combined feature count: {train.shape[1]}")

        scaler = StandardScaler()
        train_np = scaler.fit_transform(train.to_numpy())
        valid_np = scaler.transform(valid.to_numpy())
        test_np  = scaler.transform(test.to_numpy())

        train = pl.DataFrame(train_np, schema=train.columns)
        valid = pl.DataFrame(valid_np, schema=valid.columns)
        test  = pl.DataFrame(test_np,  schema=test.columns)

        joblib.dump(scaler, f"{ROOT_DIR}/src/v4/preprocessing/dimensional_reduction/scaler_v2.pkl")

        return train, valid, test
        
    def _reduce_dimension_v3(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """raw features -> pca -> pca result"""
        return self._pca(train, valid, test, version=3)
              
    def _reduce_dimension_v4(self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, latent_dim: int=32, epochs: int=20, batch_size: int=1024, lr: float=1e-3) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """raw features -> autoencoder -> ae result"""
        train, valid, test = self._autoencoder(
            train,
            valid,
            test,
            latent_dim=latent_dim,
            version=4,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr
        )

        scaler = StandardScaler()
        train_np = scaler.fit_transform(train.to_numpy())
        valid_np = scaler.transform(valid.to_numpy())
        test_np  = scaler.transform(test.to_numpy())

        train = pl.DataFrame(train_np, schema=train.columns)
        valid = pl.DataFrame(valid_np, schema=valid.columns)
        test  = pl.DataFrame(test_np,  schema=test.columns)

        joblib.dump(scaler, f"{ROOT_DIR}/src/v4/preprocessing/dimensional_reduction/scaler_v4.pkl")

        return train, valid, test
   
    def reduce_dimension(self, version: int):
        train, valid, test = self._prepare_data()
        
        self.logger.info(f'Reducing dimension v{version}...')
        
        match (version):
            case 1:
                train, valid, test = self._reduce_dimension_v1(train, valid, test)
            case 2:
                train, valid, test = self._reduce_dimension_v2(train, valid, test, latent_dim=128, epochs=100)
            case 3:
                train, valid, test = self._reduce_dimension_v3(train, valid, test)
            case 4:
                train, valid, test = self._reduce_dimension_v4(train, valid, test)
            case _:
                self.logger.error('Invalid version')
                
        self.logger.info(f"Final dimensional reduction feature count v{version}: {train.shape[1]}")
        
        train_closes = self.closes_dfs['train']
        valid_closes = self.closes_dfs['valid']
        test_closes = self.closes_dfs['test']
        
        train_temporal = self.temporal_dfs['train']
        valid_temporal = self.temporal_dfs['valid']
        test_temporal = self.temporal_dfs['test']
        
        train = pl.concat([train, train_temporal, train_closes], how="horizontal")
        valid = pl.concat([valid, valid_temporal, valid_closes], how="horizontal")
        test  = pl.concat([test, test_temporal, test_closes], how="horizontal")
        
        out_dir = f"{self.data_dir}/unified_latent"
        create_directory(out_dir)
        
        save_to_parquet(train, f"{out_dir}/unified_latent_train_v{version}.parquet")
        save_to_parquet(valid, f"{out_dir}/unified_latent_valid_v{version}.parquet")
        save_to_parquet(test,  f"{out_dir}/unified_latent_test_v{version}.parquet")
        
        self.logger.info(f"Final feature count v{version}: {train.shape[1]}")
        self.logger.info(f"Dimensionality reduction v{version} complete.")
        
        
def main():
    dimensional_reducer = DataDimensionalReducer(Logger())
    dimensional_reducer.reduce_dimension(2)
    
if __name__ == '__main__':
    main()