"""
Deep Reinforcement Learning Agent for DART
Implements the advanced RL components described in the project report.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque, namedtuple
import random
import logging
from typing import Dict, List, Tuple, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deep_rl_agent')

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'market_regime'])

class AttentionLayer(nn.Module):
    """Self-attention layer for processing temporal sequences."""
    
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection and residual connection
        output = self.out(context)
        return self.layer_norm(x + self.dropout(output))

class TradingEncoder(nn.Module):
    """Multi-modal encoder for financial data processing."""
    
    def __init__(self, technical_features=20, fundamental_features=10, sequence_length=50, d_model=128):
        super(TradingEncoder, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Technical indicators encoder
        self.technical_conv = nn.Conv1d(technical_features, d_model//2, kernel_size=3, padding=1)
        self.technical_lstm = nn.LSTM(d_model//2, d_model//2, batch_first=True, bidirectional=True)
        
        # Fundamental data encoder
        self.fundamental_linear = nn.Linear(fundamental_features, d_model//2)
        
        # Price data encoder with positional encoding
        self.price_embedding = nn.Linear(4, d_model//4)  # OHLC
        self.volume_embedding = nn.Linear(1, d_model//4)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(d_model) for _ in range(3)
        ])
        
        # Feature fusion
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, technical_data, price_data, volume_data, fundamental_data=None):
        batch_size = technical_data.size(0)
        
        # Process technical indicators
        tech_conv = F.relu(self.technical_conv(technical_data.transpose(1, 2)))
        tech_conv = tech_conv.transpose(1, 2)
        tech_lstm, _ = self.technical_lstm(tech_conv)
        
        # Process price and volume data
        price_emb = F.relu(self.price_embedding(price_data))
        volume_emb = F.relu(self.volume_embedding(volume_data.unsqueeze(-1)))
        
        # Combine price and volume embeddings
        price_volume = torch.cat([price_emb, volume_emb], dim=-1)
        
        # Concatenate all features
        combined = torch.cat([tech_lstm, price_volume], dim=-1)
        
        # Apply attention layers
        for attention in self.attention_layers:
            combined = attention(combined)
        
        # Apply dropout
        combined = self.dropout(combined)
        
        # Global average pooling over sequence
        pooled = combined.mean(dim=1)
        
        # Include fundamental data if available
        if fundamental_data is not None:
            fund_encoded = F.relu(self.fundamental_linear(fundamental_data))
            pooled = torch.cat([pooled, fund_encoded], dim=-1)
            pooled = F.relu(self.fusion_layer(pooled))
        
        return pooled

class Actor(nn.Module):
    """Actor network for continuous action spaces (position sizing)."""
    
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        # State encoder
        self.encoder = TradingEncoder()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
    def forward(self, technical_data, price_data, volume_data, fundamental_data=None):
        state_encoding = self.encoder(technical_data, price_data, volume_data, fundamental_data)
        action = self.policy_net(state_encoding)
        return self.max_action * action

class Critic(nn.Module):
    """Twin critic networks for value estimation."""
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # State encoder
        self.encoder = TradingEncoder()
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, technical_data, price_data, volume_data, action, fundamental_data=None):
        state_encoding = self.encoder(technical_data, price_data, volume_data, fundamental_data)
        state_action = torch.cat([state_encoding, action], dim=1)
        
        q1 = self.q1_net(state_action)
        q2 = self.q2_net(state_action)
        
        return q1, q2
    
    def q1(self, technical_data, price_data, volume_data, action, fundamental_data=None):
        state_encoding = self.encoder(technical_data, price_data, volume_data, fundamental_data)
        state_action = torch.cat([state_encoding, action], dim=1)
        return self.q1_net(state_action)

class MarketRegimeDetector(nn.Module):
    """Unsupervised market regime detection using VAE-like architecture."""
    
    def __init__(self, input_dim=128, latent_dim=16, n_regimes=7):
        super(MarketRegimeDetector, self).__init__()
        self.n_regimes = n_regimes
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)  # mean and log_var
        )
        
        # Regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        regime_probs = self.regime_classifier(z)
        reconstruction = self.decoder(z)
        return reconstruction, regime_probs, mean, log_var

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for more efficient learning."""
    
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done, market_regime):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done, market_regime))
        else:
            self.buffer[self.position] = Experience(state, action, reward, next_state, done, market_regime)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)

class SoftActorCritic:
    """Soft Actor-Critic (SAC) algorithm with trading-specific enhancements."""
    
    def __init__(self, state_dim=128, action_dim=3, lr_actor=3e-4, lr_critic=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.regime_detector = MarketRegimeDetector().to(self.device)
        
        # Copy parameters to target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.regime_optimizer = optim.Adam(self.regime_detector.parameters(), lr=lr_critic)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer()
        
        # Training metrics
        self.training_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'regime_loss': [],
            'alpha_loss': []
        }
    
    def select_action(self, state_data, eval_mode=False):
        """Select action using the current policy."""
        technical_data, price_data, volume_data, fundamental_data = self._process_state(state_data)
        
        with torch.no_grad():
            action = self.actor(technical_data, price_data, volume_data, fundamental_data)
            
            if eval_mode:
                return action.cpu().numpy()[0]
            else:
                # Add exploration noise during training
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1, 1)
                return action.cpu().numpy()[0]
    
    def _process_state(self, state_data):
        """Process raw state data into tensor format."""
        # Convert state data to appropriate tensor format
        # This is a placeholder - implement based on your data structure
        technical_data = torch.FloatTensor(state_data.get('technical', np.zeros((1, 50, 20)))).to(self.device)
        price_data = torch.FloatTensor(state_data.get('price', np.zeros((1, 50, 4)))).to(self.device)
        volume_data = torch.FloatTensor(state_data.get('volume', np.zeros((1, 50)))).to(self.device)
        fundamental_data = torch.FloatTensor(state_data.get('fundamental', np.zeros((1, 10)))).to(self.device) if 'fundamental' in state_data else None
        
        return technical_data, price_data, volume_data, fundamental_data
    
    def store_transition(self, state, action, reward, next_state, done, market_regime):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done, market_regime)
    
    def train(self, batch_size=256):
        """Train the agent using a batch of experiences."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(batch_size)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Unpack experiences
        states = [e.state for e in experiences]
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device).unsqueeze(1)
        next_states = [e.next_state for e in experiences]
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device).unsqueeze(1)
        
        # Process states
        current_technical, current_price, current_volume, current_fundamental = self._process_batch_states(states)
        next_technical, next_price, next_volume, next_fundamental = self._process_batch_states(next_states)
        
        # Critic loss
        with torch.no_grad():
            next_actions = self.actor(next_technical, next_price, next_volume, next_fundamental)
            noise = torch.randn_like(next_actions) * 0.1
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            next_q1, next_q2 = self.critic_target(next_technical, next_price, next_volume, next_actions, next_fundamental)
            next_q = torch.min(next_q1, next_q2) - self.alpha * torch.log(torch.abs(next_actions) + 1e-8).sum(dim=1, keepdim=True)
            target_q = rewards + self.gamma * next_q * (~dones)
        
        current_q1, current_q2 = self.critic(current_technical, current_price, current_volume, actions, current_fundamental)
        
        # Prioritized replay: compute TD errors
        td_errors1 = torch.abs(current_q1 - target_q).detach()
        td_errors2 = torch.abs(current_q2 - target_q).detach()
        td_errors = torch.max(td_errors1, td_errors2).squeeze().cpu().numpy()
        
        # Update priorities
        priorities = td_errors + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        # Weighted losses
        q1_loss = (weights * F.mse_loss(current_q1, target_q, reduction='none').squeeze()).mean()
        q2_loss = (weights * F.mse_loss(current_q2, target_q, reduction='none').squeeze()).mean()
        critic_loss = q1_loss + q2_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor loss
        current_actions = self.actor(current_technical, current_price, current_volume, current_fundamental)
        q1_actor, q2_actor = self.critic(current_technical, current_price, current_volume, current_actions, current_fundamental)
        q_actor = torch.min(q1_actor, q2_actor)
        
        actor_loss = (self.alpha * torch.log(torch.abs(current_actions) + 1e-8).sum(dim=1, keepdim=True) - q_actor).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature parameter
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (torch.log(torch.abs(current_actions) + 1e-8).sum(dim=1).detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
            self.training_metrics['alpha_loss'].append(alpha_loss.item())
        
        # Train regime detector
        regime_loss = self._train_regime_detector(current_technical, current_price, current_volume, current_fundamental)
        
        # Update target networks
        self._soft_update(self.critic, self.critic_target)
        
        # Store metrics
        self.training_metrics['actor_loss'].append(actor_loss.item())
        self.training_metrics['critic_loss'].append(critic_loss.item())
        self.training_metrics['regime_loss'].append(regime_loss)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'regime_loss': regime_loss,
            'alpha': self.alpha.item() if hasattr(self.alpha, 'item') else self.alpha
        }
    
    def _process_batch_states(self, states):
        """Process a batch of states."""
        # This is a simplified version - implement based on your state structure
        batch_size = len(states)
        technical_data = torch.stack([torch.FloatTensor(s.get('technical', np.zeros((50, 20)))) for s in states]).to(self.device)
        price_data = torch.stack([torch.FloatTensor(s.get('price', np.zeros((50, 4)))) for s in states]).to(self.device)
        volume_data = torch.stack([torch.FloatTensor(s.get('volume', np.zeros((50,)))) for s in states]).to(self.device)
        fundamental_data = None  # Add if available
        
        return technical_data, price_data, volume_data, fundamental_data
    
    def _train_regime_detector(self, technical_data, price_data, volume_data, fundamental_data):
        """Train the market regime detector."""
        # Get state encoding
        state_encoding = self.critic.encoder(technical_data, price_data, volume_data, fundamental_data)
        
        # Train regime detector
        reconstruction, regime_probs, mean, log_var = self.regime_detector(state_encoding.detach())
        
        # VAE loss
        recon_loss = F.mse_loss(reconstruction, state_encoding.detach(), reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        
        # Regime classification loss (encourage diversity)
        regime_entropy = -torch.sum(regime_probs * torch.log(regime_probs + 1e-8), dim=-1).mean()
        
        regime_loss = recon_loss + 0.1 * kl_loss - 0.01 * regime_entropy
        
        self.regime_optimizer.zero_grad()
        regime_loss.backward()
        self.regime_optimizer.step()
        
        return regime_loss.item()
    
    def _soft_update(self, source, target):
        """Soft update of target networks."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def detect_market_regime(self, state_data):
        """Detect current market regime."""
        technical_data, price_data, volume_data, fundamental_data = self._process_state(state_data)
        
        with torch.no_grad():
            state_encoding = self.critic.encoder(technical_data, price_data, volume_data, fundamental_data)
            _, regime_probs, _, _ = self.regime_detector(state_encoding)
            regime_id = torch.argmax(regime_probs, dim=-1).item()
            confidence = torch.max(regime_probs).item()
        
        regime_names = ['Bull Market', 'Bear Market', 'Sideways', 'High Volatility', 
                       'Low Volatility', 'Transition', 'Crisis']
        
        return {
            'regime_id': regime_id,
            'regime_name': regime_names[regime_id],
            'confidence': confidence,
            'probabilities': regime_probs.cpu().numpy()[0]
        }
    
    def save_models(self, filepath):
        """Save all model parameters."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'regime_detector_state_dict': self.regime_detector.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'regime_optimizer_state_dict': self.regime_optimizer.state_dict(),
            'training_metrics': self.training_metrics
        }, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath):
        """Load all model parameters."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.regime_detector.load_state_dict(checkpoint['regime_detector_state_dict'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.regime_optimizer.load_state_dict(checkpoint['regime_optimizer_state_dict'])
            
            self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
            
            logger.info(f"Models loaded from {filepath}")
            return True
        else:
            logger.warning(f"No checkpoint found at {filepath}")
            return False
    
    def get_training_metrics(self):
        """Get training metrics for monitoring."""
        return self.training_metrics.copy()
