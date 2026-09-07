import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from utils import setup_logger

logger = setup_logger("CNNLSTMModel")

class TemporalAttention(nn.Module):
    """Calculates feature-wise attention across time steps for Bidirectional LSTM."""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Hidden_Dim)
        weights = F.softmax(self.attn(x), dim=1)  # (Batch, Seq_Len, 1)
        context = torch.sum(x * weights, dim=1)   # (Batch, Hidden_Dim)
        return context

class FocalLoss(nn.Module):
    """Focal Loss to fix extreme SWANSF class imbalance."""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class SolarFlareCNNLSTM(nn.Module):
    def __init__(
        self,
        num_features=config.NUM_FEATURES,
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=config.DL_CONFIG["cnn_out_channels"],
        kernel_size=config.DL_CONFIG["cnn_kernel_size"],
        lstm_hidden_size=config.DL_CONFIG["lstm_hidden_size"],
        lstm_layers=config.DL_CONFIG["lstm_layers"],
        dropout=config.DL_CONFIG["lstm_dropout"]
    ):
        super(SolarFlareCNNLSTM, self).__init__()

        # Input Normalization Layer (Fixes unscaled physical features)
        self.input_bn = nn.BatchNorm1d(num_features)

        # Spatial Feature Extractor
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_features, cnn_out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # Deep Recurrent Encoder
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Temporal Attention Mechanism
        bidirectional_dim = lstm_hidden_size * 2
        self.attention = TemporalAttention(bidirectional_dim)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(bidirectional_dim, config.DL_CONFIG["fc_hidden"]),
            nn.BatchNorm1d(config.DL_CONFIG["fc_hidden"]),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(config.DL_CONFIG["fc_hidden"], num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Kaiming normal for Conv/Linear layers and orthogonal for LSTM."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        # x: (Batch, Channels, Seq_Len)
        x = self.input_bn(x)
        x = self.conv_block(x)          # (Batch, 128, 30)
        x = x.permute(0, 2, 1)          # (Batch, 30, 128)
        lstm_out, _ = self.lstm(x)      # (Batch, 30, 512)
        context = self.attention(lstm_out) # Attention-weighted pooling across all 30 steps
        logits = self.classifier(context)
        return logits


class CNNLSTMTrainer:
    def __init__(self, model, device=config.DEVICE, criterion=None):
        self.model = model.to(device)
        self.device = device

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = FocalLoss(gamma=2.0)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.DL_CONFIG["learning_rate"],
            weight_decay=config.DL_CONFIG["weight_decay"]
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.DL_CONFIG["epochs"], eta_min=1e-6
        )

        # Modern PyTorch AMP API
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.USE_AMP)

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Prevents exploding gradients
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        self.scheduler.step()
        return running_loss / total, correct / total

    def evaluate(self, dataloader):
        """Evaluates model strictly in eval mode with no_grad on validation/test set."""
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                probs = F.softmax(outputs, dim=1)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        return val_loss, val_acc, np.array(all_preds), np.array(all_targets), np.array(all_probs)

    def predict_proba(self, dataloader):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs = inputs.to(self.device)
                with torch.amp.autocast('cuda', enabled=config.USE_AMP):
                    outputs = self.model(inputs)
                    probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_probs)

    def save_checkpoint(self, filename="best_solar_flare_model.pt"):
        filepath = os.path.join(config.MODEL_SAVE_DIR, filename)
        torch.save(self.model.state_dict(), filepath)
