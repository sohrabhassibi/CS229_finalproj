#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

SEQ_LEN = 30
BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("data/SHEL_data.csv")
feature_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
values = df[feature_cols].astype(np.float32).values
n_samples, n_features = values.shape

split_idx = int(0.9 * n_samples)
scaler = StandardScaler()
values_train = values[:split_idx]
scaler.fit(values_train)
values_scaled = scaler.transform(values).astype(np.float32)

def make_sequences(values_scaled, seq_len, split_idx):
    X_train, y_train = [], []
    X_test, y_test = [], []
    for t in range(seq_len, split_idx):
        X_train.append(values_scaled[t-seq_len:t])
        y_train.append(values_scaled[t])
    n_total = values_scaled.shape[0]
    for t in range(split_idx, n_total):
        if t - seq_len < 0:
            continue
        X_test.append(values_scaled[t-seq_len:t])
        y_test.append(values_scaled[t])
    return np.stack(X_train), np.stack(y_train), np.stack(X_test), np.stack(y_test)

X_train, y_train, X_test, y_test = make_sequences(values_scaled, SEQ_LEN, split_idx)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class SimpleMultiheadAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x_proj = self.input_proj(x)
        attn_out, attn_weights = self.attn(x_proj, x_proj, x_proj)
        last_hidden = attn_out[:, -1]
        pred = self.fc(last_hidden)
        return pred, attn_weights

class MLPForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super().__init__()
        flattened_size = seq_len * input_dim
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        x = self.relu(self.fc1(x_flat))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        pred = self.fc4(x)
        return pred

class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        flattened_size = SEQ_LEN * input_dim
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.backcast = nn.Linear(hidden_dim, flattened_size)
        self.forecast = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x_flat):
        x = self.relu(self.fc1(x_flat))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        backcast = self.backcast(x)
        forecast = self.forecast(x)
        return backcast, forecast

class NBeatsForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_stacks=2, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_stacks):
            for _ in range(num_blocks):
                self.blocks.append(NBeatsBlock(input_dim, hidden_dim, output_dim))
    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        forecast_sum = 0
        for block in self.blocks:
            backcast, forecast = block(x_flat)
            forecast_sum = forecast_sum + forecast
            x_flat = x_flat - backcast
        return forecast_sum

class TCNForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_levels=3, kernel_size=3):
        super().__init__()
        layers = []
        num_channels = [hidden_dim] * num_levels
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]
        pred = self.fc(y)
        return pred

class MoiraiBaseline(nn.Module):
    def forward(self, x):
        return x[:, -1, :].contiguous()


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_metrics(y_true_scaled, y_pred_scaled, X_test_scaled_last):
    mse_scaled = np.mean((y_pred_scaled - y_true_scaled) ** 2)
    rmse_scaled = np.sqrt(mse_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    eps = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
    last = X_test_scaled_last
    actual_change = np.sign(y_true_scaled - last)
    pred_change = np.sign(y_pred_scaled - last)
    mda = (actual_change == pred_change).astype(np.float32).mean()
    r2 = r2_score(y_true_scaled.reshape(-1), y_pred_scaled.reshape(-1))
    bias_per_feature = np.mean(y_pred - y_true, axis=0)
    bias_overall = bias_per_feature.mean()
    avg_abs_level = np.mean(np.abs(y_true))
    threshold = 0.01 * avg_abs_level
    if abs(bias_overall) < threshold:
        bias_flag = "None / minimal"
    elif bias_overall > 0:
        bias_flag = "Over-forecast (too high)"
    else:
        bias_flag = "Under-forecast (too low)"
    return {
        "Scaled_RMSE": float(rmse_scaled),
        "MAPE_percent": float(mape),
        "MDA": float(mda),
        "R2": float(r2),
        "Bias_overall": float(bias_overall),
        "Bias_flag": bias_flag,
    }

def train_model(model, model_name, train_loader, num_epochs=30):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.contiguous().to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{num_epochs}, Train MSE: {train_loss/len(train_loader.dataset):.4f}")
    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_preds_scaled = []
    all_true_scaled = []
    all_last_inputs_scaled = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.contiguous().to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            if isinstance(pred, tuple):
                pred = pred[0]
            all_preds_scaled.append(pred.cpu().numpy())
            all_true_scaled.append(yb.cpu().numpy())
            all_last_inputs_scaled.append(xb[:, -1, :].cpu().numpy())
    return (np.concatenate(all_preds_scaled, axis=0),
            np.concatenate(all_true_scaled, axis=0),
            np.concatenate(all_last_inputs_scaled, axis=0))

def plot_overlay(y_true, y_pred, model_name, save_path):
    n_test = y_true.shape[0]
    t_axis = np.arange(n_test)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]
    for d, col in enumerate(feature_cols):
        ax = axes[d]
        ax.plot(t_axis, y_true[:, d], label=f"True {col}", linewidth=2)
        ax.plot(t_axis, y_pred[:, d], label=f"Pred {col}", linewidth=2, alpha=0.7)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Test window index (time)")
    plt.suptitle(f"{model_name}: True vs Pred on Holdout (last 10%)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_all_features_separate(y_true, y_pred, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for d, col in enumerate(feature_cols):
        fig, ax = plt.subplots(figsize=(12, 6))
        t_axis = np.arange(len(y_true))
        ax.plot(t_axis, y_true[:, d], label=f"True {col}", linewidth=2, alpha=0.8)
        ax.plot(t_axis, y_pred[:, d], label=f"Pred {col}", linewidth=2, alpha=0.8)
        ax.set_ylabel(col, fontsize=12)
        ax.set_xlabel("Test window index", fontsize=12)
        ax.set_title(f"{model_name}: {col} - True vs Predicted", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        rmse = np.sqrt(mean_squared_error(y_true[:, d], y_pred[:, d]))
        ax.text(0.02, 0.98, f"RMSE: {rmse:.4f}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name.replace(' ', '_')}_{col.replace(' ', '_')}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()

def ablation_study(model, X_test, y_test, feature_cols):
    model.eval()
    baseline_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH_SIZE):
            batch_x = torch.from_numpy(X_test[i:i+BATCH_SIZE]).float().contiguous().to(DEVICE)
            pred = model(batch_x)
            if isinstance(pred, tuple):
                pred = pred[0]
            baseline_preds.append(pred.cpu().numpy())
    baseline_preds = np.concatenate(baseline_preds, axis=0)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    ablation_results = {}
    for feat_idx, feat_name in enumerate(feature_cols):
        X_test_ablated = X_test.copy()
        X_test_ablated[:, :, feat_idx] = 0
        ablated_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test_ablated), BATCH_SIZE):
                batch_x = torch.from_numpy(X_test_ablated[i:i+BATCH_SIZE]).float().contiguous().to(DEVICE)
                pred = model(batch_x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                ablated_preds.append(pred.cpu().numpy())
        ablated_preds = np.concatenate(ablated_preds, axis=0)
        ablated_rmse = np.sqrt(mean_squared_error(y_test, ablated_preds))
        impact = ablated_rmse - baseline_rmse
        ablation_results[feat_name] = {
            'baseline_rmse': baseline_rmse,
            'ablated_rmse': ablated_rmse,
            'impact': impact,
            'impact_pct': (impact / baseline_rmse) * 100
        }
    return ablation_results, baseline_rmse

def plot_ablation_results(ablation_results, baseline_rmse, model_name, save_path):
    features = list(ablation_results.keys())
    impacts = [ablation_results[f]['impact'] for f in features]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x > 0 else 'green' for x in impacts]
    ax.barh(features, impacts, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('RMSE Increase (higher = more important)', fontsize=12)
    ax.set_title(f'{model_name}: Ablation Study Results', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_feature_attribution(model, X_test_sample):
    model.eval()
    X_baseline = torch.from_numpy(X_test_sample).float().contiguous().to(DEVICE)
    with torch.no_grad():
        baseline_pred = model(X_baseline)
        if isinstance(baseline_pred, tuple):
            baseline_pred = baseline_pred[0]
        baseline_pred = baseline_pred.cpu().numpy()
    attribution_map = np.zeros((SEQ_LEN, n_features))
    for t in range(SEQ_LEN):
        for f in range(n_features):
            X_perturbed = X_test_sample.copy()
            X_perturbed[0, t, f] = 0
            X_pert_tensor = torch.from_numpy(X_perturbed).float().contiguous().to(DEVICE)
            with torch.no_grad():
                perturbed_pred = model(X_pert_tensor)
                if isinstance(perturbed_pred, tuple):
                    perturbed_pred = perturbed_pred[0]
                perturbed_pred = perturbed_pred.cpu().numpy()
            attribution = np.mean(np.abs(baseline_pred - perturbed_pred))
            attribution_map[t, f] = attribution
    return attribution_map

def plot_attribution_heatmap(attribution_map, feature_cols, model_name, save_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    attribution_norm = attribution_map / (attribution_map.max() + 1e-8)
    im = ax.imshow(attribution_norm.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(np.arange(SEQ_LEN))
    ax.set_xticklabels([f"t-{SEQ_LEN-i-1}" for i in range(SEQ_LEN)])
    ax.set_yticks(np.arange(len(feature_cols)))
    ax.set_yticklabels(feature_cols)
    ax.set_xlabel("Time Step (lookback)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"{model_name}: Feature Attribution Map", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Attribution Score (normalized)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def extract_and_plot_attention(model, X_test_sample, feature_cols, model_name, save_path):
    model.eval()
    X_tensor = torch.from_numpy(X_test_sample).float().contiguous().to(DEVICE)
    with torch.no_grad():
        output = model(X_tensor)
        if isinstance(output, tuple) and len(output) == 2:
            pred, attn_weights = output
            attn_weights = attn_weights.cpu().numpy()
            if len(attn_weights.shape) == 4:
                attn_weights = attn_weights.mean(axis=1)
            if len(attn_weights.shape) == 3:
                attn_weights = attn_weights[0]
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(attn_weights, aspect='auto', cmap='Blues', interpolation='nearest')
            ax.set_xticks(np.arange(SEQ_LEN))
            ax.set_xticklabels([f"t-{SEQ_LEN-i-1}" for i in range(SEQ_LEN)])
            ax.set_yticks(np.arange(SEQ_LEN))
            ax.set_yticklabels([f"t-{SEQ_LEN-i-1}" for i in range(SEQ_LEN)])
            ax.set_xlabel("Key/Value (source)", fontsize=12)
            ax.set_ylabel("Query (target)", fontsize=12)
            ax.set_title(f"{model_name}: Attention Weight Heatmap", fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label="Attention Weight")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return True
    return False

train_ds = TimeSeriesDataset(X_train, y_train)
test_ds = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

def analyze_model(model_class, model_name, model_kwargs, has_attention=False):
    
    model = model_class(**model_kwargs).to(DEVICE)
    print(f"Training {model_name}...")
    model = train_model(model, model_name, train_loader, NUM_EPOCHS)
    
    print(f"Evaluating {model_name}...")
    y_pred_scaled, y_true_scaled, X_last_scaled = evaluate_model(model, test_loader)
    
    metrics = compute_metrics(y_true_scaled, y_pred_scaled, X_last_scaled)
    
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)
    
    output_dir = f"results/{model_name.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating plots for {model_name}")
    overlay_path = f"{output_dir}/overlay.png"
    plot_overlay(y_true, y_pred, model_name, overlay_path)
    plot_all_features_separate(y_true, y_pred, model_name, f"{output_dir}/features")
    
    print(f"Running explainability analysis for {model_name}")
    ablation_results, baseline_rmse = ablation_study(model, X_test, y_test, feature_cols)
    plot_ablation_results(ablation_results, baseline_rmse, model_name, f"{output_dir}/ablation.png")
    
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx+1]
    attribution_map = compute_feature_attribution(model, sample)
    plot_attribution_heatmap(attribution_map, feature_cols, model_name, f"{output_dir}/attribution_map.png")
    
    if has_attention:
        extract_and_plot_attention(model, sample, feature_cols, model_name, f"{output_dir}/attention_heatmap.png")
    
    print(f"\n{model_name} Metrics:")
    print(f"  Scaled RMSE: {metrics['Scaled_RMSE']:.6f}")
    print(f"  MAPE: {metrics['MAPE_percent']:.2f}%")
    print(f"  MDA: {metrics['MDA']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    print(f"  Forecast Bias: {metrics['Bias_flag']}")
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'overlay_path': overlay_path
    }

print("\n[4/6] Running model analysis")

all_results = []

results_simple_attn = analyze_model(
    SimpleMultiheadAttn,
    "Simple Multihead Attn",
    {'input_dim': n_features, 'hidden_dim': HIDDEN_DIM, 'output_dim': n_features},
    has_attention=True
)
all_results.append(results_simple_attn)

results_mlp = analyze_model(
    MLPForecaster,
    "Open Source MLP",
    {'input_dim': n_features, 'hidden_dim': HIDDEN_DIM, 'output_dim': n_features, 'seq_len': SEQ_LEN},
    has_attention=False
)
all_results.append(results_mlp)

results_nbeats = analyze_model(
    NBeatsForecaster,
    "N-beats",
    {'input_dim': n_features, 'hidden_dim': HIDDEN_DIM, 'output_dim': n_features},
    has_attention=False
)
all_results.append(results_nbeats)

results_tcn = analyze_model(
    TCNForecaster,
    "TCN",
    {'input_dim': n_features, 'hidden_dim': HIDDEN_DIM, 'output_dim': n_features},
    has_attention=False
)
all_results.append(results_tcn)

results_morai = analyze_model(
    MoiraiBaseline,
    "Moirai",
    {},
    has_attention=False
)
all_results.append(results_morai)

summary_data = []
for res in all_results:
    m = res['metrics']
    summary_data.append({
        'Model': res['model_name'],
        'Scaled RMSE': m['Scaled_RMSE'],
        'MAPE': m['MAPE_percent'],
        'Mean Directional Accuracy (MDA)': m['MDA'],
        'R^2': m['R2'],
        'Forecast Bias?': m['Bias_flag'],
        'Overlay Plot (link)': res['overlay_path']
    })

summary_df = pd.DataFrame(summary_data)
print("SUMMARY")
print(summary_df.to_string(index=False))

summary_df.to_csv("model_results_summary.csv", index=False)
