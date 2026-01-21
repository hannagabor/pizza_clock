import torch as t
from torch import Tensor, nn
from jaxtyping import Float, Int
import einops
from pizza_clock.config import Config


class Model(nn.Module):
    # One-layer RELU transformer with bidirectional attention regulated by the attention rate.

    # For a transformer with “width” d, the input embedding and the residue stream will be d-dimensional,
    # 4 attention heads of ⌊d/4⌋ dimensions will be employed, and the MLP will be of 4d hidden units.
    # By default d = 128 is chosen. ReLU is used as the activation function and layer normalization isn’t
    # applied. The post-softmax attention matrix is interpolated between an all-one matrix and original as
    # specified by the attention rate. TODO: this is a bit strange, maybe try dividing the all-one matrix
    # by p.

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        t.manual_seed(self.config.seed)

        self.token_embedding = EmbeddingLayer(
            config.p, config.residual_dim, device=self.config.device
        )
        self.position_embedding = EmbeddingLayer(
            2, config.residual_dim, device=self.config.device
        )

        self.num_attention_heads = 4
        self.head_dim = config.residual_dim // self.num_attention_heads
        self.attention_rate = config.attention_rate

        self.num_mlp_hidden_units = 4 * config.residual_dim

        self.attention = Attention(
            config.attention_rate,
            n_heads=self.num_attention_heads,
            d_model=config.residual_dim,
            d_head=self.head_dim,
            init_range=0.02,
            device=self.config.device,
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.residual_dim, self.num_mlp_hidden_units),
            nn.ReLU(),
            nn.Linear(self.num_mlp_hidden_units, config.residual_dim),
        )
        self.fc1 = nn.Linear(config.residual_dim, self.num_mlp_hidden_units)
        self.fc2 = nn.Linear(self.num_mlp_hidden_units, config.residual_dim)
        self.unembedding = nn.Linear(config.residual_dim, config.p)

    def forward(
        self, x: Int[Tensor, "batch position token"]
    ) -> Float[Tensor, "batch position vocab"]:
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(
            t.tensor([0, 1]).to(self.config.device)
        )
        x = token_embeddings + position_embeddings

        x = self.attention(x)
        x = t.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        logits = self.unembedding(x)
        return logits


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, device: t.device):
        super().__init__()
        self.weight = t.randn(vocab_size, embedding_dim, device=device) / (
            embedding_dim**0.5
        )

    def forward(
        self, x: Int[Tensor, "batch position token"]
    ) -> Float[Tensor, "batch position embedding_dim"]:
        return self.weight[x]


class Attention(nn.Module):
    # Adapted from ARENA

    def __init__(
        self,
        attention_rate: float,
        n_heads: int,
        d_model: int,
        d_head: int,
        init_range: float,
        device: t.device,
    ):
        super().__init__()
        self.attention_rate = attention_rate
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head
        self.init_range = init_range
        self.device = device
        self.W_Q = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_K = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_V = nn.Parameter(t.empty((n_heads, d_model, d_head)))
        self.W_O = nn.Parameter(t.empty((n_heads, d_head, d_model)))
        self.b_Q = nn.Parameter(t.zeros((n_heads, d_head)))
        self.b_K = nn.Parameter(t.zeros((n_heads, d_head)))
        self.b_V = nn.Parameter(t.zeros((n_heads, d_head)))
        self.b_O = nn.Parameter(t.zeros((d_model)))
        nn.init.normal_(self.W_Q, std=self.init_range)
        nn.init.normal_(self.W_K, std=self.init_range)
        nn.init.normal_(self.W_V, std=self.init_range)
        nn.init.normal_(self.W_O, std=self.init_range)

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        ) / (self.d_head**0.5)
        attn_pattern = attn_scores.softmax(-1)

        attn_pattern = (1 - self.attention_rate) + self.attention_rate * attn_pattern

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return attn_out
