import torch
import torch.nn as nn



class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        # features: (N, T, D)
        pooled = features.mean(dim=1)   # token mean pooling, size (N, D)
        h = self.ln(pooled)
        h = self.fc1(h)
        h = self.act(h)
        logit = self.fc2(h)
        return logit.squeeze(-1)


class BCE_Loss(nn.Module):
    def __init__(self, detach_features=False, reduction="mean"):
        super().__init__()
        self.detach_features = detach_features
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, classifier, features, labels):
        # features: (N, T, D), labels: (N,) or (N, 1)
        if self.detach_features:
            features = features.detach()

        logits = classifier(features)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        labels = labels.to(device=logits.device, dtype=logits.dtype)

        return self.criterion(logits, labels)
