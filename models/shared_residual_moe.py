import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from models.sample_style_sparse_moge import Attention, GCN
except ModuleNotFoundError:
    from sample_style_sparse_moge import Attention, GCN


class SharedEncoderBlock(nn.Module):
    def __init__(self, hidden_channels, num_points, heads, dim_head, dropout=0.2):
        super().__init__()
        dim = hidden_channels * num_points
        self.num_points = num_points
        self.attention = nn.Sequential(
            Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
            nn.LayerNorm(dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.gcn = GCN(hidden_channels, hidden_channels)
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, adjacency):
        y = self.attention(x) + x
        y_graph = rearrange(y, "N T (P F) -> N T P F", P=self.num_points)
        y_graph = self.gcn(y_graph, adjacency) + y_graph
        y = rearrange(y_graph, "N T P F -> N T (P F)")
        return self.out(y)


class SharedEEGEncoder(nn.Module):
    def __init__(
        self,
        in_channels=5,
        hidden_channels=64,
        num_points=62,
        time_window=5,
        num_layers=1,
        heads=2,
        dim_head=4,
        pool="cls",
        dropout=0.2,
        nonnegative_adjacency=False,
    ):
        super().__init__()
        assert pool in ["cls", "mean"], "pool must be 'cls' or 'mean'"

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_points = num_points
        self.pool = pool
        self.nonnegative_adjacency = nonnegative_adjacency

        input_dim = in_channels * num_points
        hidden_dim = hidden_channels * num_points

        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, time_window + 1, input_dim))
        self.ln = nn.LayerNorm(input_dim)
        self.embedder = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                SharedEncoderBlock(
                    hidden_channels=hidden_channels,
                    num_points=num_points,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        xs, ys = torch.tril_indices(num_points, num_points, offset=-1)
        self.register_buffer("xs", xs, persistent=False)
        self.register_buffer("ys", ys, persistent=False)
        adjacency = torch.empty(num_points, num_points)
        nn.init.uniform_(adjacency)
        self.A = nn.Parameter(adjacency[xs, ys], requires_grad=True)
        self.output_dim = hidden_dim

    def build_adjacency(self, device):
        adjacency = torch.zeros((self.num_points, self.num_points), device=device)
        edge_weights = F.softplus(self.A) if self.nonnegative_adjacency else self.A
        adjacency[self.xs, self.ys] = edge_weights
        adjacency = adjacency + adjacency.T
        adjacency = adjacency + torch.eye(self.num_points, device=device)
        return adjacency

    def forward(self, x):
        # x: [N, T, 5, 62]
        N, T, V, C = x.shape
        if V != self.in_channels or C != self.num_points:
            raise ValueError("Expected x shape [N, T, {}, {}], got {}".format(
                self.in_channels, self.num_points, tuple(x.shape)
            ))

        x = x.view(N, T, V * C)
        cls_tokens = repeat(self.cls_token, "() T D -> N T D", N=N)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        T = T + 1
        x = self.ln(x.view(N * T, V * C)).view(N, T, V, C)
        x = rearrange(x, "N T V C -> N T C V")
        x = self.embedder(x)
        x = rearrange(x, "N T C F -> N T (C F)")

        adjacency = self.build_adjacency(x.device)
        for layer in self.layers:
            x = layer(x, adjacency)

        if self.pool == "mean":
            return x.mean(dim=1)
        return x[:, 0]


class StyleTopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k=2, temperature=1.0, dropout=0.2):
        super().__init__()
        assert top_k >= 1, "top_k must be >= 1"
        assert top_k <= num_experts, "top_k must be <= num_experts"
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        hidden_dim = max(dim // 2, 1)
        self.router = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, z_shared):
        logits = self.router(z_shared) / self.temperature
        topk_values, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_values, dim=-1)
        router_weights = torch.zeros_like(logits)
        router_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)
        return router_weights


class ResidualExpert(nn.Module):
    def __init__(self, dim, bottleneck_dim=None, dropout=0.2):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(dim // 4, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )

    def forward(self, z_shared):
        return self.net(z_shared)


class ResidualMoEBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_experts=4,
        top_k=2,
        temperature=1.0,
        dropout=0.2,
        expert_bottleneck=None,
    ):
        super().__init__()
        self.router = StyleTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature,
            dropout=dropout,
        )
        self.experts = nn.ModuleList(
            [
                ResidualExpert(dim=dim, bottleneck_dim=expert_bottleneck, dropout=dropout)
                for _ in range(num_experts)
            ]
        )

    def forward(self, z_shared):
        router_weights = self.router(z_shared)
        residual = 0.0
        for expert_index, expert in enumerate(self.experts):
            expert_residual = expert(z_shared)
            weight = router_weights[:, expert_index].view(z_shared.size(0), 1)
            residual = residual + weight * expert_residual
        z_final = z_shared + residual
        return z_final, router_weights


class SharedResidualMoGE(nn.Module):
    def __init__(
        self,
        in_channels=5,
        hidden_channels=64,
        num_points=62,
        time_window=5,
        num_layers=1,
        heads=2,
        dim_head=4,
        num_classes=3,
        num_experts=4,
        top_k=2,
        temperature=1.0,
        pool="cls",
        dropout=0.2,
        expert_bottleneck=None,
        nonnegative_adjacency=False,
    ):
        super().__init__()
        dim = hidden_channels * num_points
        if expert_bottleneck is None:
            expert_bottleneck = max(dim // 4, 1)

        self.shared_encoder = SharedEEGEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_points=num_points,
            time_window=time_window,
            num_layers=num_layers,
            heads=heads,
            dim_head=dim_head,
            pool=pool,
            dropout=dropout,
            nonnegative_adjacency=nonnegative_adjacency,
        )
        self.residual_moe = ResidualMoEBlock(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            temperature=temperature,
            dropout=dropout,
            expert_bottleneck=expert_bottleneck,
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, return_router=False):
        z_shared = self.shared_encoder(x)
        z_final, router_weights = self.residual_moe(z_shared)
        logits = self.classifier(z_final)

        if return_router:
            return logits, [router_weights]
        return logits


if __name__ == "__main__":
    model = SharedResidualMoGE(
        in_channels=5,
        hidden_channels=64,
        num_points=62,
        time_window=5,
        num_layers=1,
        heads=2,
        dim_head=4,
        num_classes=4,
        num_experts=4,
        top_k=2,
        temperature=1.0,
        pool="cls",
        dropout=0.2,
    )

    x = torch.rand(32, 5, 5, 62)
    logits, router_weights_all = model(x, return_router=True)

    print(logits.shape)
    print(router_weights_all[0].shape)
