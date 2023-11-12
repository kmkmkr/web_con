import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyValueNet(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.convt1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        self.convt2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.convt3 = nn.ConvTranspose2d(32, out_shape[0], kernel_size=8, stride=4)
        self.fc = nn.Linear(64*25*25, 512)
        self.fc_v = nn.Linear(512, 1)

        self.mask = torch.zeros((210,160), dtype=torch.int16)
        self.mask = F.pad(self.mask, (37, 37, 12, 12), mode='constant', value=1).long()
        self.mask = self.mask.reshape(-1)
    def extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
    def policy_net(self, x):
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))
        return x
    def forward(self, x):
        x = self.extract_features(x)
        policy = self.policy_net(x)
        value = F.relu(self.fc_v(F.relu(self.fc(x.flatten(start_dim=1)))))
        return policy, value
    
    def predict_values(self, x):
        x = self.extract_features(x)
        value = F.relu(self.fc_v(F.relu(self.fc(x.flatten(start_dim=1)))))
        return value
    
    @staticmethod
    def log_prob(value, logits):
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        log_prob = log_pmf.gather(-1, value).squeeze(-1)
        return log_prob

    @staticmethod
    def entropy(logits):
        logits = logits.flatten(start_dim=1)
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        probs = F.softmax(logits, dim=-1)
        p_log_p = logits * probs
        return -p_log_p.sum(-1)

    def sample(self, obs):
        logits, values = self.forward(obs)
        logits = logits.flatten(start_dim=1)
        # Normalize
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        probs[:,self.mask] = 0
        actions = torch.multinomial(probs, 1, True)
        return actions, values, self.log_prob(actions, logits)
    
    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        logits = logits.flatten(start_dim=1)
        # Normalize
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        log_prob = self.log_prob(actions, logits)
        entropy = self.entropy(logits)
        return values, log_prob, entropy