"""
Deep Reinforcement Learning Agent for DART v2.0
Enhanced with advanced SAC implementation, n-step returns, curiosity-driven exploration,
and automatic entropy tuning.
"""

import logging
import math
import os
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deep_rl_agent")

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done", "market_regime"],
)

# N-step experience for multi-step returns
NStepExperience = namedtuple(
    "NStepExperience", ["states", "actions", "rewards", "next_state", "done", "market_regime"],
)


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
    """Multi-modal encoder for financial data processing - v2.0 enhanced."""

    def __init__(
        self, technical_features=20, fundamental_features=10, sequence_length=50, d_model=128,
    ):
        super(TradingEncoder, self).__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length

        # Technical indicators encoder
        self.technical_conv = nn.Conv1d(technical_features, d_model // 2, kernel_size=3, padding=1)
        self.technical_lstm = nn.LSTM(
            d_model // 2,
            d_model // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.1,
        )

        # Fundamental data encoder
        self.fundamental_linear = nn.Linear(fundamental_features, d_model // 2)

        # Price data encoder with positional encoding
        self.price_embedding = nn.Linear(4, d_model // 4)  # OHLC
        self.volume_embedding = nn.Linear(1, d_model // 4)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(sequence_length, d_model)

        # Attention layers - increased depth for v2.0
        self.attention_layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(4)])

        # Feature fusion
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(0.1)

    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, technical_data, price_data, volume_data, fundamental_data=None):
        _batch_size = technical_data.size(0)  # noqa: F841 - kept for debugging

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

        # Add positional encoding
        seq_len = combined.size(1)
        combined = combined + self.positional_encoding[:, :seq_len, : combined.size(-1)]

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


class GaussianActor(nn.Module):
    """Gaussian actor network for SAC with reparameterization trick."""

    def __init__(self, state_dim, action_dim, max_action=1.0, log_std_min=-20, log_std_max=2):
        super(GaussianActor, self).__init__()
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # State encoder
        self.encoder = TradingEncoder()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Mean and log_std heads
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, technical_data, price_data, volume_data, fundamental_data=None):
        state_encoding = self.encoder(technical_data, price_data, volume_data, fundamental_data)
        shared_features = self.shared(state_encoding)

        mean = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, technical_data, price_data, volume_data, fundamental_data=None):
        """Sample action using reparameterization trick."""
        mean, log_std = self.forward(technical_data, price_data, volume_data, fundamental_data)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample

        # Apply tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action * self.max_action, log_prob, mean

    def get_action(
        self, technical_data, price_data, volume_data, fundamental_data=None, deterministic=False,
    ):
        """Get action for evaluation."""
        mean, log_std = self.forward(technical_data, price_data, volume_data, fundamental_data)

        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)

        return action * self.max_action


class Critic(nn.Module):
    """Twin critic networks for value estimation - v2.0 enhanced."""

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # State encoder (shared for efficiency)
        self.encoder = TradingEncoder()

        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(128 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
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


class CuriosityModule(nn.Module):
    """Intrinsic Curiosity Module (ICM) for exploration - v2.0 feature."""

    def __init__(self, state_dim=128, action_dim=3, feature_dim=64):
        super(CuriosityModule, self).__init__()

        # Inverse model: predict action from state and next_state
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        # Forward model: predict next_state feature from state and action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

        # State feature encoder
        self.feature_encoder = nn.Sequential(nn.Linear(state_dim, feature_dim), nn.ReLU())

        self.feature_dim = feature_dim

    def forward(self, state, next_state, action):
        """Compute intrinsic reward from curiosity."""
        # Encode states to features
        _state_feat = self.feature_encoder(state)  # noqa: F841 - features learned via inverse model
        next_state_feat = self.feature_encoder(next_state)

        # Inverse model prediction (for learning state features)
        combined_states = torch.cat([state, next_state], dim=-1)
        pred_action = self.inverse_model(combined_states)

        # Forward model prediction
        state_action = torch.cat([state, action], dim=-1)
        pred_next_feat = self.forward_model(state_action)

        # Intrinsic reward: prediction error of forward model
        intrinsic_reward = 0.5 * F.mse_loss(
            pred_next_feat, next_state_feat.detach(), reduction="none",
        ).mean(dim=-1)

        # Inverse model loss
        inverse_loss = F.mse_loss(pred_action, action)

        # Forward model loss
        forward_loss = F.mse_loss(pred_next_feat, next_state_feat.detach())

        return intrinsic_reward, inverse_loss, forward_loss


class MarketRegimeDetector(nn.Module):
    """Enhanced market regime detection using VAE architecture - v2.0."""

    def __init__(self, input_dim=128, latent_dim=16, n_regimes=7):
        super(MarketRegimeDetector, self).__init__()
        self.n_regimes = n_regimes
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim * 2),  # mean and log_var
        )

        # Regime classifier with temperature scaling
        self.regime_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, n_regimes),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

        # Regime embeddings for conditioning
        self.regime_embeddings = nn.Embedding(n_regimes, latent_dim)

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

        # Temperature-scaled softmax for regime classification
        regime_logits = self.regime_classifier(z)
        regime_probs = F.softmax(regime_logits / self.temperature, dim=-1)

        reconstruction = self.decoder(z)
        return reconstruction, regime_probs, mean, log_var, z

    def get_regime_embedding(self, regime_ids):
        return self.regime_embeddings(regime_ids)


class NStepReplayBuffer:
    """N-step prioritized experience replay buffer for multi-step returns - v2.0 feature."""

    def __init__(
        self, capacity=100000, n_steps=3, gamma=0.99, alpha=0.6, beta=0.4, beta_increment=0.001,
    ):
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

        # N-step buffer for computing returns
        self.n_step_buffer = deque(maxlen=n_steps)

    def _get_n_step_info(self):
        """Calculate n-step return and get final next state."""
        reward, next_state, done = (
            self.n_step_buffer[-1][2],
            self.n_step_buffer[-1][3],
            self.n_step_buffer[-1][4],
        )

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, next_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = next_s, d

        return reward, next_state, done

    def add(self, state, action, reward, next_state, done, market_regime):
        self.n_step_buffer.append((state, action, reward, next_state, done, market_regime))

        if len(self.n_step_buffer) < self.n_steps:
            return

        # Get n-step return
        n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()

        # Get first state and action
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        market_regime = self.n_step_buffer[0][5]

        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(
                Experience(
                    state, action, n_step_reward, n_step_next_state, n_step_done, market_regime,
                ),
            )
        else:
            self.buffer[self.position] = Experience(
                state, action, n_step_reward, n_step_next_state, n_step_done, market_regime,
            )

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.position]

        probs = priorities**self.alpha
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


class SoftActorCriticV2:
    """
    Soft Actor-Critic v2.0 with trading-specific enhancements:
    - Automatic entropy tuning with improved target entropy
    - N-step returns for better credit assignment
    - Curiosity-driven exploration
    - Gradient clipping and regularization
    - Market regime-aware learning
    """

    def __init__(
        self,
        state_dim=128,
        action_dim=3,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        n_steps=3,
        use_curiosity=True,
        curiosity_coef=0.01,
        target_entropy_scale=1.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"DART v2.0 Deep RL Agent using device: {self.device}")

        self.gamma = gamma
        self.tau = tau
        self.n_steps = n_steps
        self.use_curiosity = use_curiosity
        self.curiosity_coef = curiosity_coef
        self.action_dim = action_dim

        # Networks
        self.actor = GaussianActor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.regime_detector = MarketRegimeDetector().to(self.device)

        # Curiosity module
        if use_curiosity:
            self.curiosity = CuriosityModule(state_dim, action_dim).to(self.device)
            self.curiosity_optimizer = optim.Adam(self.curiosity.parameters(), lr=lr_critic)

        # Copy parameters to target network
        self._hard_update(self.critic, self.critic_target)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_critic, weight_decay=1e-5,
        )
        self.regime_optimizer = optim.Adam(self.regime_detector.parameters(), lr=lr_critic)

        # Automatic entropy tuning with improved target
        self.target_entropy = -action_dim * target_entropy_scale
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
        self.alpha = self.log_alpha.exp().detach()

        # N-step replay buffer
        self.replay_buffer = NStepReplayBuffer(n_steps=n_steps, gamma=gamma)

        # Training metrics
        self.training_metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "regime_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "entropy": [],
            "intrinsic_reward": [],
            "q_value": [],
        }

        self.training_steps = 0

        logger.info("SAC v2.0 initialized with n-step returns and curiosity exploration")

    def _hard_update(self, source, target):
        """Hard update of target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state_data, eval_mode=False):
        """Select action using the current policy."""
        technical_data, price_data, volume_data, fundamental_data = self._process_state(state_data)

        with torch.no_grad():
            action = self.actor.get_action(
                technical_data, price_data, volume_data, fundamental_data, deterministic=eval_mode,
            )
            return action.cpu().numpy()[0]

    def _process_state(self, state_data):
        """Process raw state data into tensor format."""
        technical_data = torch.FloatTensor(state_data.get("technical", np.zeros((1, 50, 20)))).to(
            self.device,
        )
        price_data = torch.FloatTensor(state_data.get("price", np.zeros((1, 50, 4)))).to(
            self.device,
        )
        volume_data = torch.FloatTensor(state_data.get("volume", np.zeros((1, 50)))).to(self.device)
        fundamental_data = (
            torch.FloatTensor(state_data.get("fundamental", np.zeros((1, 10)))).to(self.device)
            if "fundamental" in state_data
            else None
        )

        return technical_data, price_data, volume_data, fundamental_data

    def store_transition(self, state, action, reward, next_state, done, market_regime):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done, market_regime)

    def train(self, batch_size=256):
        """Train the agent using a batch of experiences with v2.0 enhancements."""
        if len(self.replay_buffer) < batch_size:
            return None

        self.training_steps += 1

        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(batch_size)
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)

        # Unpack experiences
        states = [e.state for e in experiences]
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device).unsqueeze(1)
        next_states = [e.next_state for e in experiences]
        dones = torch.FloatTensor([float(e.done) for e in experiences]).to(self.device).unsqueeze(1)

        # Process states
        current_technical, current_price, current_volume, current_fundamental = (
            self._process_batch_states(states)
        )
        next_technical, next_price, next_volume, next_fundamental = self._process_batch_states(
            next_states,
        )

        # Get state encodings for curiosity
        with torch.no_grad():
            current_encoding = self.critic.encoder(
                current_technical, current_price, current_volume, current_fundamental,
            )
            next_encoding = self.critic.encoder(
                next_technical, next_price, next_volume, next_fundamental,
            )

        # Compute intrinsic reward if using curiosity
        intrinsic_reward = torch.zeros_like(rewards)
        if self.use_curiosity:
            intrinsic_reward, inverse_loss, forward_loss = self.curiosity(
                current_encoding, next_encoding, actions,
            )
            intrinsic_reward = intrinsic_reward.unsqueeze(1) * self.curiosity_coef

            # Update curiosity module
            curiosity_loss = inverse_loss + forward_loss
            self.curiosity_optimizer.zero_grad()
            curiosity_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.curiosity.parameters(), 1.0)
            self.curiosity_optimizer.step()

        # Total reward = extrinsic + intrinsic
        total_rewards = rewards + intrinsic_reward.detach()

        # Update critic with n-step returns (gamma^n already computed in buffer)
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(
                next_technical, next_price, next_volume, next_fundamental,
            )

            next_q1, next_q2 = self.critic_target(
                next_technical, next_price, next_volume, next_actions, next_fundamental,
            )
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs

            # N-step target with proper discounting
            target_q = total_rewards + (self.gamma**self.n_steps) * next_q * (1 - dones)

        current_q1, current_q2 = self.critic(
            current_technical, current_price, current_volume, actions, current_fundamental,
        )

        # Compute TD errors for prioritized replay
        td_errors1 = torch.abs(current_q1 - target_q).detach()
        td_errors2 = torch.abs(current_q2 - target_q).detach()
        td_errors = torch.max(td_errors1, td_errors2).squeeze().cpu().numpy()

        # Update priorities
        priorities = td_errors + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

        # Weighted Huber loss for robustness
        critic_loss = (weights * F.smooth_l1_loss(current_q1, target_q, reduction="none")).mean()
        critic_loss += (weights * F.smooth_l1_loss(current_q2, target_q, reduction="none")).mean()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update actor (delayed update for stability)
        if self.training_steps % 2 == 0:
            current_actions, log_probs, _ = self.actor.sample(
                current_technical, current_price, current_volume, current_fundamental,
            )

            q1_actor, q2_actor = self.critic(
                current_technical,
                current_price,
                current_volume,
                current_actions,
                current_fundamental,
            )
            q_actor = torch.min(q1_actor, q2_actor)

            actor_loss = (self.alpha * log_probs - q_actor).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Update alpha (entropy temperature)
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            self.training_metrics["alpha_loss"].append(alpha_loss.item())
            self.training_metrics["actor_loss"].append(actor_loss.item())
            self.training_metrics["entropy"].append(-log_probs.mean().item())

        # Train regime detector
        regime_loss = self._train_regime_detector(current_encoding.detach())

        # Soft update target networks
        self._soft_update(self.critic, self.critic_target)

        # Store metrics
        self.training_metrics["critic_loss"].append(critic_loss.item())
        self.training_metrics["regime_loss"].append(regime_loss)
        self.training_metrics["alpha"].append(self.alpha.item())
        self.training_metrics["intrinsic_reward"].append(intrinsic_reward.mean().item())
        self.training_metrics["q_value"].append(current_q1.mean().item())

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": self.training_metrics["actor_loss"][-1]
            if self.training_metrics["actor_loss"]
            else 0,
            "regime_loss": regime_loss,
            "alpha": self.alpha.item(),
            "intrinsic_reward": intrinsic_reward.mean().item(),
            "q_value": current_q1.mean().item(),
        }

    def _process_batch_states(self, states):
        """Process a batch of states."""
        _batch_size = len(states)  # noqa: F841 - kept for documentation
        technical_data = torch.stack(
            [torch.FloatTensor(s.get("technical", np.zeros((50, 20)))) for s in states],
        ).to(self.device)
        price_data = torch.stack(
            [torch.FloatTensor(s.get("price", np.zeros((50, 4)))) for s in states],
        ).to(self.device)
        volume_data = torch.stack(
            [torch.FloatTensor(s.get("volume", np.zeros((50,)))) for s in states],
        ).to(self.device)
        fundamental_data = None

        return technical_data, price_data, volume_data, fundamental_data

    def _train_regime_detector(self, state_encoding):
        """Train the market regime detector."""
        reconstruction, regime_probs, mean, log_var, z = self.regime_detector(state_encoding)

        # VAE loss with KL annealing
        recon_loss = F.mse_loss(reconstruction, state_encoding, reduction="mean")
        kl_weight = min(1.0, self.training_steps / 10000)  # KL annealing
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        # Regime diversity loss (encourage using all regimes)
        regime_entropy = -torch.sum(regime_probs * torch.log(regime_probs + 1e-8), dim=-1).mean()

        regime_loss = recon_loss + kl_weight * 0.1 * kl_loss - 0.01 * regime_entropy

        self.regime_optimizer.zero_grad()
        regime_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.regime_detector.parameters(), 1.0)
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
            state_encoding = self.critic.encoder(
                technical_data, price_data, volume_data, fundamental_data,
            )
            _, regime_probs, _, _, z = self.regime_detector(state_encoding)
            regime_id = torch.argmax(regime_probs, dim=-1).item()
            confidence = torch.max(regime_probs).item()

        regime_names = [
            "Bull Market",
            "Bear Market",
            "Sideways",
            "High Volatility",
            "Low Volatility",
            "Transition",
            "Crisis",
        ]

        return {
            "regime_id": regime_id,
            "regime_name": regime_names[regime_id],
            "confidence": confidence,
            "probabilities": regime_probs.cpu().numpy()[0],
            "latent_representation": z.cpu().numpy()[0],
        }

    def get_uncertainty(self, state_data):
        """Get action uncertainty for exploration/exploitation trade-off."""
        technical_data, price_data, volume_data, fundamental_data = self._process_state(state_data)

        with torch.no_grad():
            mean, log_std = self.actor(technical_data, price_data, volume_data, fundamental_data)
            std = log_std.exp()
            uncertainty = std.mean().item()

        return {
            "uncertainty": uncertainty,
            "action_mean": mean.cpu().numpy()[0],
            "action_std": std.cpu().numpy()[0],
        }

    def save_models(self, filepath):
        """Save all model parameters."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "regime_detector_state_dict": self.regime_detector.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "regime_optimizer_state_dict": self.regime_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "training_metrics": self.training_metrics,
            "training_steps": self.training_steps,
        }

        if self.use_curiosity:
            checkpoint["curiosity_state_dict"] = self.curiosity.state_dict()
            checkpoint["curiosity_optimizer_state_dict"] = self.curiosity_optimizer.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"SAC v2.0 models saved to {filepath}")

    def load_models(self, filepath):
        """Load all model parameters."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)

            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.regime_detector.load_state_dict(checkpoint["regime_detector_state_dict"])

            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.regime_optimizer.load_state_dict(checkpoint["regime_optimizer_state_dict"])

            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

            self.training_metrics = checkpoint.get("training_metrics", self.training_metrics)
            self.training_steps = checkpoint.get("training_steps", 0)

            if self.use_curiosity and "curiosity_state_dict" in checkpoint:
                self.curiosity.load_state_dict(checkpoint["curiosity_state_dict"])
                self.curiosity_optimizer.load_state_dict(
                    checkpoint["curiosity_optimizer_state_dict"],
                )

            logger.info(f"SAC v2.0 models loaded from {filepath}")
            return True
        else:
            logger.warning(f"No checkpoint found at {filepath}")
            return False

    def get_training_metrics(self):
        """Get training metrics for monitoring."""
        return self.training_metrics.copy()


# Alias for backward compatibility
DeepRLAgent = SoftActorCriticV2
SoftActorCritic = SoftActorCriticV2


# Legacy support
class PrioritizedReplayBuffer(NStepReplayBuffer):
    """Legacy alias for backward compatibility."""

    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(
            capacity=capacity,
            n_steps=1,
            gamma=0.99,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment,
        )


# Actor alias for backward compatibility
Actor = GaussianActor
