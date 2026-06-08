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


class SampleStyleTopKRouter(nn.Module):
    """
    Sample-style top-k sparse router
    鏍锋湰椋庢牸 top-k 绋€鐤忚矾鐢卞櫒

    鏍稿績鎬濇兂锛?    - 涓嶄娇鐢?subject_id锛?    - 鏍规嵁 EEG 鏍锋湰鑷韩鐨勫叏灞€椋庢牸鐗瑰緛鐢熸垚 expert 鏉冮噸锛?    - 姣忎釜鏍锋湰鍙縺娲?top-k 涓?expert锛?    - 淇濈暀 MoE / GMoE 鐨?sparse expert 鎬濇兂銆?
    杈撳叆:
        y: [N, T, P, F]
           N = batch size
           T = time steps
           P = EEG channels / graph nodes
           F = hidden feature dimension

    杈撳嚭:
        sparse_weights: [N, E]
           E = num_experts
    """

    def __init__(
        self,
        hidden_dim,
        num_experts,
        top_k=2,
        temperature=1.0,
        dropout=0.1,
    ):
        super().__init__()

        assert top_k >= 1, "top_k must be >= 1"
        assert top_k <= num_experts, "top_k must be <= num_experts"

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, y):
        # y: [N, T, P, F]

        # 瀵规椂闂寸淮搴?T 鍜岄€氶亾/鑺傜偣缁村害 P 鍋氬钩鍧囨睜鍖?        # 寰楀埌姣忎釜鏍锋湰鐨?EEG 椋庢牸鍚戦噺
        # style: [N, F]
        style = y.mean(dim=(1, 2))

        # logits: [N, E]
        logits = self.router(style)

        # 娓╁害缂╂斁
        logits = logits / self.temperature

        # 鍙?top-k expert
        topk_values, topk_indices = torch.topk(
            logits,
            k=self.top_k,
            dim=-1,
        )

        # 鍙湪 top-k expert 鍐呭仛 softmax
        topk_weights = torch.softmax(topk_values, dim=-1)

        # Build sparse routing weights over experts.
        sparse_weights = torch.zeros_like(logits)
        sparse_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)

        return sparse_weights


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
        dim_head,
        num_experts,
        top_k=2,
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

        # 鏍锋湰椋庢牸 top-k 绋€鐤忚矾鐢卞櫒
        self.router = SampleStyleTopKRouter(
            hidden_dim=in_channels,
            num_experts=num_experts,
            top_k=top_k,
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

        # temporal attention
        self.attention = nn.Sequential(
            Attention(
                in_channels * num_points,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            ),
            nn.LayerNorm(in_channels * num_points),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        # causal temporal convolution
        self.pad = nn.ZeroPad2d((2, 0, 0, 0))
        self.causal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            stride=1,
        )

        self.squ = nn.Sequential(
            nn.LayerNorm(out_channels * num_points),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.last_router_weights = None

    def forward(self, x, A, return_router=False):
        """
        杈撳叆:
            x: [N, T, P * F]
               P = num_points
               F = in_channels

            A: [P, P]

        杈撳嚭:
            y: [N, T, P * Fout]
        """

        # temporal attention
        y = self.attention(x) + x

        # [N, T, P*F] -> [N, T, P, F]
        y = rearrange(
            y,
            "N T (P F) -> N T P F",
            P=self.num_points,
        )

        N, T, P, F = y.shape

        # 鏍锋湰绾ч鏍艰矾鐢?        # router_weights: [N, E]
        if self.router_mode == "single_expert":
            router_weights = y.new_zeros((N, self.num_experts))
            router_weights[:, self.fixed_expert_index] = 1.0
        else:
            router_weights = self.router(y)
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

        # [N, T, P, Fout] -> [(N*P), Fout, T]
        y = rearrange(y, "N T P F -> (N P) F T")

        # causal temporal convolution
        y = self.pad(y)
        y = y.unsqueeze(1)      # [(N*P), 1, Fout, T+2]
        y = self.causal_conv(y) # [(N*P), 1, Fout, T]
        y = y.squeeze(1)        # [(N*P), Fout, T]

        # [(N*P), Fout, T] -> [N, T, P*Fout]
        y = rearrange(
            y,
            "(N P) F T -> N T (P F)",
            N=N,
            P=P,
        )

        y = self.squ(y)

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

        assert pool in ["cls", "mean"], "pool must be 'cls' or 'mean'"
        assert top_k <= num_experts, "top_k must be <= num_experts"

        self.num_points = num_points
        self.pool = pool
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.nonnegative_adjacency = nonnegative_adjacency
        self.router_mode = router_mode
        self.fixed_expert_index = fixed_expert_index

        dim = in_channels * num_points

        self.pos_embedding = nn.Parameter(
            torch.randn(1, time_window + 1, dim)
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, dim)
        )

        self.ln = nn.LayerNorm(dim)

        # 灏嗛娈电淮搴?in_channels 鏄犲皠鍒?hidden_channels
        self.embedder = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                SampleStyleSparseMoeGCNTransformerUnit(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_points=num_points,
                    heads=heads,
                    dim_head=dim_head,
                    num_experts=num_experts,
                    top_k=top_k,
                    temperature=temperature,
                    dropout=dropout,
                    router_mode=router_mode,
                    fixed_expert_index=fixed_expert_index,
                )
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(hidden_channels * num_points, num_classes)

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

        # [N, T, V, C] -> [N, T, V*C]
        x = x.view(N, T, V * C)

        # add cls token
        cls_tokens = repeat(
            self.cls_token,
            "() T D -> N T D",
            N=N,
        )

        x = torch.cat((cls_tokens, x), dim=1)

        # add position embedding
        x = x + self.pos_embedding[:, : T + 1]

        T = T + 1

        # LayerNorm
        x = x.view(N * T, V * C)
        x = self.ln(x)
        x = x.view(N, T, V, C)

        # [N, T, V, C] -> [N, T, C, V]
        # C 鏄?EEG channels锛孷 鏄?frequency bands
        x = rearrange(x, "N T V C -> N T C V")

        # 瀵规瘡涓?EEG channel 鐨勯娈电壒寰佸仛 embedding
        # [N, T, C, V] -> [N, T, C, hidden_channels]
        x = self.embedder(x)

        # [N, T, C, hidden] -> [N, T, C*hidden]
        x = rearrange(x, "N T C F -> N T (C F)")

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

        if self.pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        features = x
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

