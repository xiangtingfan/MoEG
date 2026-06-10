import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


#########################################################
# Sample-style Sparse MoGE
# 鏍锋湰椋庢牸鎰熺煡绋€鐤忓浘涓撳娣峰悎妯″瀷
#
# 鏀瑰姩鐩爣锛?# 1. 淇濈暀鍘熷 MoGE 涓诲共缁撴瀯锛?# 2. 涓嶉澶栧姞鍏?shared emotion branch锛?# 3. 灏嗗師鏉ョ殑 channel-level hard routing
#    鏀逛负 sample-style top-k sparse routing锛?# 4. 姣忎釜 EEG 鏍锋湰鏍规嵁鑷韩椋庢牸閫夋嫨 top-k 涓?GCN experts锛?# 5. router 涓嶄娇鐢?subject_id锛屽彧鏍规嵁 EEG 琛ㄥ緛鏈韩璺敱銆?#########################################################


class GCN(nn.Module):
    """
    Graph Convolutional Network
    鍥惧嵎绉綉缁?
    杈撳叆:
        X: [N, T, P, Fin]
           N   = batch size
           T   = time steps
           P   = EEG channels / graph nodes
           Fin = input feature dimension

        A: [P, P]
           adjacency matrix
           閭绘帴鐭╅樀

    杈撳嚭:
        output: [N, T, P, Fout]
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def norm_adjacency(self, A):
        """
        Symmetric normalized adjacency:
        D^{-1/2} A D^{-1/2}
        """
        d = A.sum(1)
        d_inv_sqrt = torch.pow(d + 1e-8, -0.5)
        D = torch.diag(d_inv_sqrt)
        return D.mm(A).mm(D)

    def forward(self, X, A):
        # X: [N, T, P, Fin]
        # weight: [Fin, Fout]
        support = torch.einsum("ntpf,fo->ntpo", X, self.weight)

        norm_A = self.norm_adjacency(A)

        # norm_A: [P, P]
        # support: [N, T, P, Fout]
        output = torch.einsum("qp,ntpo->ntqo", norm_A, support)

        if self.use_bias:
            output = output + self.bias

        return output


class Attention(nn.Module):
    """
    Multi-head Self-Attention
    澶氬ご鑷敞鎰忓姏
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        # x: [N, T, D]
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            qkv,
        )

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class ChannelWiseTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, T, P, H]
        N, T, P, H = x.shape
        x_ = x.permute(0, 2, 1, 3).reshape(N * P, T, H)
        attn_out, attn_weight = self.attn(x_, x_, x_, need_weights=True)
        x_ = self.norm(x_ + self.dropout(attn_out))
        x_ = x_.reshape(N, P, T, H).permute(0, 2, 1, 3)
        return x_, attn_weight


class SampleLevelRouter(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_points,
        num_experts,
        top_k=2,
        router_dim=256,
        temperature=1.0,
        dropout=0.2,
    ):
        super().__init__()
        assert top_k >= 1, "top_k must be >= 1"
        assert top_k <= num_experts, "top_k must be <= num_experts"

        self.top_k = top_k
        self.temperature = temperature
        self.num_experts = num_experts

        in_dim = hidden_dim * num_points
        self.time_score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, router_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(router_dim, num_experts),
        )

    def forward(self, y):
        # y: [N, T, P, H]
        N, T, P, H = y.shape
        y_flat = y.reshape(N, T, P * H)
        score = self.time_score(y_flat)
        alpha = torch.softmax(score, dim=1)
        z = (y_flat * alpha).sum(dim=1)
        logits = self.router(z) / self.temperature

        topk_values, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_values, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)
        return weights, alpha


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_points, dropout=0.1):
        super().__init__()
        in_dim = hidden_dim * num_points
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 4, 1),
        )

    def forward(self, y):
        # y: [N, T, P, H]
        N, T, P, H = y.shape
        y_flat = y.reshape(N, T, P * H)
        score = self.score(y_flat)
        alpha = torch.softmax(score, dim=1)
        z = (y_flat * alpha).sum(dim=1)
        return z, alpha


class SampleStyleSparseMoeGCNTransformerUnit(nn.Module):
    """
    Sample-style Sparse MoGE block
    鏍锋湰椋庢牸鎰熺煡绋€鐤?MoGE 妯″潡

    涓庡師濮?Moe_GCN_Transformer_unit 鐨勬牳蹇冨尯鍒細
    鍘熷鐗堟湰:
        channel / local position -> argmax -> one expert

    褰撳墠鐗堟湰:
        sample EEG style -> top-k sparse routing -> weighted experts

    涔熷氨鏄锛?        涓嶆槸姣忎釜閫氶亾鍗曠嫭閫?expert锛?        鑰屾槸姣忎釜鏍锋湰鏍规嵁鏁翠綋 EEG 椋庢牸閫夋嫨灏戦噺 expert銆?    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_points,
        heads,
        num_experts,
        top_k=2,
        router_dim=256,
        temperature=1.0,
        dropout=0.2,
        router_mode="learned",
        fixed_expert_index=0,
    ):
        super().__init__()

        self.num_points = num_points
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_mode = router_mode
        self.fixed_expert_index = fixed_expert_index
        if router_mode not in {"learned", "single_expert"}:
            raise ValueError("router_mode must be learned or single_expert")
        if not 0 <= fixed_expert_index < num_experts:
            raise ValueError("fixed_expert_index must be in [0, num_experts)")

        self.router = SampleLevelRouter(
            hidden_dim=in_channels,
            num_points=num_points,
            num_experts=num_experts,
            top_k=top_k,
            router_dim=router_dim,
            temperature=temperature,
            dropout=dropout,
        )

        # GCN experts
        self.GCNs = nn.ModuleList(
            [
                GCN(in_channels, out_channels)
                for _ in range(num_experts)
            ]
        )

        # 濡傛灉杈撳叆杈撳嚭缁村害涓嶅悓锛屼娇鐢ㄧ嚎鎬ф姇褰卞仛 residual
        if in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = nn.Identity()

        self.out_norm = nn.Sequential(
            nn.LayerNorm(out_channels * num_points),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.last_router_weights = None

    def forward(self, x, A, return_router=False):
        """
        杈撳叆:
            x: [N, T, P, F]
               P = num_points
               F = in_channels

            A: [P, P]

        杈撳嚭:
            y: [N, T, P, Fout]
        """
        y = x
        N, T, P, F = y.shape

        # 鏍锋湰绾ч鏍艰矾鐢?        # router_weights: [N, E]
        if self.router_mode == "single_expert":
            router_weights = y.new_zeros((N, self.num_experts))
            router_weights[:, self.fixed_expert_index] = 1.0
        else:
            router_weights, _ = self.router(y)
        self.last_router_weights = router_weights.detach()

        # residual path
        residual = self.res_proj(y)

        # sparse expert fusion
        # 铏界劧杩欓噷浠ｇ爜涓婇亶鍘嗘墍鏈?expert锛屼絾闈?top-k expert 鐨勬潈閲嶄负 0
        # 鍥犳鏁板涓婁粛鏄?sparse routing
        moe_out = 0.0

        for e in range(self.num_experts):
            expert_out = self.GCNs[e](y, A)  # [N, T, P, Fout]

            # 褰撳墠 expert 瀵规瘡涓牱鏈殑鏉冮噸
            w_e = router_weights[:, e].view(N, 1, 1, 1)

            moe_out = moe_out + w_e * expert_out

        # MoGE 涓诲共寮忔洿鏂帮細expert 杈撳嚭 + residual
        y = residual + moe_out

        y_flat = rearrange(y, "N T P F -> N T (P F)")
        y_flat = self.out_norm(y_flat)
        y = rearrange(y_flat, "N T (P F) -> N T P F", P=P)

        if return_router:
            return y, router_weights

        return y


class SampleStyleSparseMoGE(nn.Module):
    """
    Sample-style Sparse MoGE
    鏍锋湰椋庢牸鎰熺煡绋€鐤忓浘涓撳娣峰悎妯″瀷

    杈撳叆 x:
        x.shape = [N, T, V, C]

        N = batch size
        T = time window
        V = frequency bands
        C = EEG channels

    渚嬪瓙锛?        SEED / SEED-IV DE 鐗瑰緛:
        x = [batch, time_window, 5, 62]
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_points,
        time_window,
        num_layers,
        heads,
        dim_head,
        num_classes,
        num_experts=6,
        top_k=2,
        temperature=1.0,
        pool="cls",
        dropout=0.2,
        nonnegative_adjacency=False,
        router_mode="learned",
        fixed_expert_index=0,
    ):
        super().__init__()

        assert top_k <= num_experts, "top_k must be <= num_experts"

        self.num_points = num_points
        self.pool = pool
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.nonnegative_adjacency = nonnegative_adjacency
        self.router_mode = router_mode
        self.fixed_expert_index = fixed_expert_index

        # 灏嗛娈电淮搴?in_channels 鏄犲皠鍒?hidden_channels
        self.embedder = nn.Linear(in_channels, hidden_channels)
        self.channel_temporal_attn = ChannelWiseTemporalAttention(
            hidden_dim=hidden_channels,
            num_heads=heads,
            dropout=dropout,
        )

        self.layers = nn.ModuleList(
            [
                SampleStyleSparseMoeGCNTransformerUnit(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_points=num_points,
                    heads=heads,
                    num_experts=num_experts,
                    top_k=top_k,
                    router_dim=256,
                    temperature=temperature,
                    dropout=dropout,
                    router_mode=router_mode,
                    fixed_expert_index=fixed_expert_index,
                )
                for _ in range(num_layers)
            ]
        )

        self.temporal_pool = TemporalAttentionPooling(
            hidden_dim=hidden_channels,
            num_points=num_points,
            dropout=dropout,
        )
        classifier_in_dim = hidden_channels * num_points
        classifier_hidden_dim = 256
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(classifier_in_dim),
            nn.Linear(classifier_in_dim, classifier_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(classifier_hidden_dim, num_classes)

        # learnable adjacency matrix
        self.xs, self.ys = torch.tril_indices(
            self.num_points,
            self.num_points,
            offset=-1,
        )

        adjacency = torch.Tensor(self.num_points, self.num_points)
        nn.init.uniform_(adjacency)

        self.A = nn.Parameter(
            adjacency[self.xs, self.ys],
            requires_grad=True,
        )

    def build_adjacency(self, device):
        adjacency = torch.zeros(
            (self.num_points, self.num_points),
            device=device,
        )

        edge_weights = F.softplus(self.A) if self.nonnegative_adjacency else self.A
        adjacency[self.xs, self.ys] = edge_weights
        adjacency = adjacency + adjacency.T

        # 鍔犺嚜杩炴帴
        adjacency = adjacency + torch.eye(self.num_points, device=device)

        return adjacency

    def forward(self, x, return_router=False, return_features=False):
        """
        杈撳叆:
            x: [N, T, V, C]
               N = batch size
               T = time window
               V = frequency bands
               C = EEG channels

        杈撳嚭:
            out: [N, num_classes]

        濡傛灉 return_router=True:
            杩斿洖:
                out, router_weights_all

            router_weights_all:
                list锛屾瘡灞備竴涓?[N, num_experts]
        """

        N, T, V, C = x.shape

        assert C == self.num_points, (
            f"Expected EEG channels C={self.num_points}, but got C={C}"
        )

        # [N, T, V, C] -> [N, T, C, V]
        # C 鏄?EEG channels锛孷 鏄?frequency bands
        x = rearrange(x, "N T V C -> N T C V")

        # 瀵规瘡涓?EEG channel 鐨勯娈电壒寰佸仛 embedding
        # [N, T, C, V] -> [N, T, C, hidden_channels]
        x = self.embedder(x)
        x, _ = self.channel_temporal_attn(x)

        adjacency = self.build_adjacency(x.device)

        router_weights_all = []

        for layer in self.layers:
            if return_router:
                x, router_weights = layer(
                    x,
                    adjacency,
                    return_router=True,
                )
                router_weights_all.append(router_weights)
            else:
                x = layer(x, adjacency)

        x, _ = self.temporal_pool(x)

        features = self.classifier_head(x)
        out = self.fc(features)

        if return_router and return_features:
            return out, router_weights_all, features
        if return_router:
            return out, router_weights_all
        if return_features:
            return out, features

        return out


#########################################################
# Routing regularization losses
# 璺敱姝ｅ垯椤?#########################################################


#########################################################
# Example usage
# 浣跨敤绀轰緥
#########################################################

if __name__ == "__main__":
    from losses.router_losses import router_balance_loss, router_entropy_loss

    model = SampleStyleSparseMoGE(
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
    )

    # x.shape = [N, T, V, C]
    # N: batch size
    # T: time window
    # V: frequency bands
    # C: EEG channels
    x = torch.rand(size=(32, 5, 5, 62))

    logits, router_weights_all = model(x, return_router=True)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)

    for i, w in enumerate(router_weights_all):
        print(f"Layer {i} router weights shape:", w.shape)
        print(f"Layer {i} first 3 samples routing weights:")
        print(w[:3])

    labels = torch.randint(0, 3, size=(32,))
    criterion = nn.CrossEntropyLoss()

    loss_cls = criterion(logits, labels)
    loss_bal = router_balance_loss(router_weights_all)
    loss_ent = router_entropy_loss(router_weights_all)

    loss = loss_cls + 0.05 * loss_bal + 0.01 * loss_ent

    print("loss_cls:", loss_cls.item())
    print("loss_bal:", loss_bal.item())
    print("loss_ent:", loss_ent.item())
    print("total loss:", loss.item())

