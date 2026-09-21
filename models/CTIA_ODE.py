import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class RevIN(nn.Module):

    def __init__(
        self,
        num_features,
        eps=1e-5,
        affine=True
    ):

        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:

            self.affine_weight = nn.Parameter(
                torch.ones(num_features)
            )

            self.affine_bias = nn.Parameter(
                torch.zeros(num_features)
            )

    def forward(
        self,
        x,
        mode
    ):

        if mode == "norm":

            self.mean = x.mean(
                dim=-1,
                keepdim=True
            ).detach()

            self.stdev = torch.sqrt(
                x.var(
                    dim=-1,
                    keepdim=True,
                    unbiased=False
                )
                +
                self.eps
            ).detach()

            x = (
                x - self.mean
            ) / self.stdev

            if self.affine:

                x = (
                    x
                    *
                    self.affine_weight.view(
                        1,
                        -1,
                        1
                    )
                )

                x = (
                    x
                    +
                    self.affine_bias.view(
                        1,
                        -1,
                        1
                    )
                )

            return x

        elif mode == "denorm":

            if self.affine:

                x = (
                    x
                    -
                    self.affine_bias.view(
                        1,
                        -1,
                        1
                    )
                )

                x = x / (
                    self.affine_weight.view(
                        1,
                        -1,
                        1
                    )
                    +
                    self.eps
                )

            x = (
                x * self.stdev
            ) + self.mean

            return x

        else:

            raise ValueError(
                "mode must be norm or denorm"
            )

class VariableEncoder(nn.Module):

    def __init__(
        self,
        seq_len,
        d_model,
        dropout=0.05
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                seq_len,
                d_model * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model * 2,
                d_model
            ),

            nn.LayerNorm(
                d_model
            )
        )

    def forward(self, x):

        return self.encoder(x)

class AsymmetricPrimaryFusion(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.05
    ):

        super().__init__()

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.gate = nn.Sequential(

            nn.Linear(
                d_model * 2,
                d_model
            ),

            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.scale = math.sqrt(
            d_model
        )

    def forward(
        self,
        H,
        primary_idx
    ):

        # H: (B, M, D)

        B, M, D = H.shape

        if M <= 1:

            return H

        aux_indices = [
            i
            for i in range(M)
            if i != primary_idx
        ]

        H_primary = H[
            :,
            primary_idx:
            primary_idx + 1,
            :
        ]

        H_aux = H[
            :,
            aux_indices,
            :
        ]

        Q = self.q_proj(
            H_primary
        )

        K = self.k_proj(
            H_aux
        )

        V = self.v_proj(
            H_aux
        )

        scores = torch.matmul(
            Q,
            K.transpose(-1, -2)
        ) / self.scale

        attention = F.softmax(
            scores,
            dim=-1
        )

        context = torch.matmul(
            attention,
            V
        )

        gate = self.gate(
            torch.cat(
                [
                    H_primary,
                    context
                ],
                dim=-1
            )
        )

        H_primary_new = self.norm(
            H_primary
            +
            gate
            *
            self.dropout(context)
        )

        H_out = H.clone()

        H_out[
            :,
            primary_idx:
            primary_idx + 1,
            :
        ] = H_primary_new

        return H_out

class CTIAODEFunc(nn.Module):

    def __init__(
        self,
        d_model
    ):

        super().__init__()

        self.d_model = d_model

        self.state_net = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * 2
            ),

            nn.Tanh(),

            nn.Linear(
                d_model * 2,
                d_model
            )
        )

        nn.init.zeros_(self.state_net[-1].weight)
        nn.init.zeros_(self.state_net[-1].bias)

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.alpha = nn.Parameter(
            torch.tensor(0.1)
        )

        self.K_exo = None
        self.V_exo = None

    def update_exogenous(
        self,
        K_exo,
        V_exo
    ):

        self.K_exo = K_exo
        self.V_exo = V_exo

    def forward(
        self,
        t,
        H
    ):

        # --------------------------------------------------------
        # Base dynamics
        # --------------------------------------------------------

        dH = self.state_net(
            H
        )

        # --------------------------------------------------------
        # Continuous-time cross attention
        # --------------------------------------------------------

        if (
            self.K_exo is not None
            and
            self.V_exo is not None
        ):

            Q = self.q_proj(
                H
            )

            scores = torch.matmul(
                Q,
                self.K_exo.transpose(
                    -1,
                    -2
                )
            )

            scores = scores / math.sqrt(
                self.d_model
            )

            attention = F.softmax(
                scores,
                dim=-1
            )

            context = torch.matmul(
                attention,
                self.V_exo
            )

            dH = (
                dH
                +
                self.alpha * context
            )

        return dH

class LoopedStableBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads=4,
        max_loops=100,
        dropout=0.05
    ):

        super().__init__()

        self.max_loops = max_loops

        self.epsilon = 1e-3

        self.attn = nn.MultiheadAttention(

            embed_dim=d_model,

            num_heads=num_heads,

            dropout=dropout,

            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.ffn = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model * 2,
                d_model
            )
        )

    def forward(self, H):

        H_current = H

        for _ in range(
            self.max_loops
        ):

            attn_out, _ = self.attn(

                H_current,

                H_current,

                H_current,

                need_weights=False
            )

            H_new = self.norm1(

                H_current

                +

                attn_out
            )

            H_new = self.norm2(

                H_new

                +

                self.ffn(H_new)
            )

            diff = torch.norm(
                H_new - H_current,
                dim=-1
            ).mean()

            base = torch.norm(
                H_current,
                dim=-1
            ).mean()

            relative_diff = (
                    diff /
                    (base + 1e-6)
            )

            H_current = H_new

            if (
                    relative_diff.item()
                    < self.epsilon
            ):

                break

        return H_current



class MLPForecastHead(nn.Module):

    def __init__(
        self,
        d_model,
        pred_len,
        dropout=0.05
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * 2
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_model * 2,
                pred_len
            )
        )


    def forward(self, H):

        return self.net(H)

class FutureTimeModule(nn.Module):

    def __init__(
        self,
        num_time_features,
        d_model
    ):

        super().__init__()

        self.time_encoder = nn.Sequential(

            nn.Linear(
                num_time_features,
                d_model
            ),

            nn.GELU(),

            nn.Linear(
                d_model,
                d_model
            )
        )

        self.variable_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.time_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.scale = math.sqrt(
            d_model
        )

    def forward(
        self,
        H,
        future_time
    ):

        # --------------------------------------------------------
        # future_time:
        # (B, F, T)
        #
        # -> (B, T, F)
        # --------------------------------------------------------

        future_time = future_time.transpose(
            1,
            2
        )

        # --------------------------------------------------------
        # (B, T, D)
        # --------------------------------------------------------

        time_hidden = self.time_encoder(
            future_time
        )

        H_var = self.variable_proj(
            H
        )

        H_time = self.time_proj(
            time_hidden
        )

        # --------------------------------------------------------
        # (B, M, D)
        # x
        # (B, T, D)
        #
        # ->
        #
        # (B, M, T)
        # --------------------------------------------------------

        time_effect = torch.einsum(
            "bmd,btd->bmt",
            H_var,
            H_time
        )

        time_effect = (
            time_effect
            /
            self.scale
        )

        return time_effect


class MovingAverage(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x shape: (B, M, L)
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=-1)

        x_trend = self.avg(x_pad)
        x_seasonal = x - x_trend

        return x_trend, x_seasonal

class CTIAODEForecaster(nn.Module):

    def __init__(
        self,
        seq_len,
        pred_len,
        num_vars,
        num_time_features,
        d_model=128,
        primary_idx=-1,
        dropout=0.05
    ):

        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_vars = num_vars
        self.d_model = d_model

        self.primary_idx = primary_idx



        self.revin = RevIN(
            num_vars
        )



        self.variable_encoder = VariableEncoder(

            seq_len=seq_len,

            d_model=d_model,

            dropout=dropout
        )



        self.asymmetric_fusion = (
            AsymmetricPrimaryFusion(

                d_model=d_model,

                dropout=dropout
            )
        )


        self.k_proj = nn.Linear(

            d_model,

            d_model,

            bias=False
        )

        self.v_proj = nn.Linear(

            d_model,

            d_model,

            bias=False
        )


        self.ode_func = CTIAODEFunc(
            d_model
        )



        self.stable_block = LoopedStableBlock(

            d_model=d_model,

            num_heads=4,

            max_loops=2,

            dropout=dropout
        )


        self.forecast_head = MLPForecastHead(

            d_model=d_model,

            pred_len=pred_len,

            dropout=dropout
        )


        self.future_time_module = (
            FutureTimeModule(

                num_time_features=num_time_features,

                d_model=d_model
            )
        )

        self.decomp = MovingAverage(kernel_size=25)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)


        self.beta_time = nn.Parameter(
            torch.tensor(0.05)
        )

        self.beta_linear = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(
        self,
        x,
        history_time,
        future_time
    ):

        B, M, L = x.shape


        if self.primary_idx == -1:

            primary_idx = M - 1

        else:

            primary_idx = self.primary_idx


        x_norm = self.revin(
            x,
            "norm"
        )


        H = self.variable_encoder(
            x_norm
        )


        H = self.asymmetric_fusion(
            H,
            primary_idx
        )


        aux_indices = [

            i
            for i in range(M)

            if i != primary_idx
        ]

        H_aux = H[
            :,
            aux_indices,
            :
        ]


        K_exo = self.k_proj(
            H_aux
        )

        V_exo = self.v_proj(
            H_aux
        )

        self.ode_func.update_exogenous(
            K_exo,
            V_exo
        )


        t_span = torch.linspace(

            0.0,

            1.0,

            steps=5,

            dtype=x.dtype,

            device=x.device
        )

        ode_out = odeint(

            self.ode_func,

            H,

            t_span,

            method="rk4"
        )

        H_T = ode_out[-1]


        H_stable = self.stable_block(
            H_T
        )


        nonlinear_pred = self.forecast_head(
            H_stable
        )

        time_effect = self.future_time_module(

            H_stable,

            future_time
        )

        trend, seasonal = self.decomp(x_norm)
        linear_pred = self.linear_trend(trend) + self.linear_seasonal(seasonal)


        pred_norm = (

            nonlinear_pred

            +

            self.beta_time
            *
            time_effect

            +

            self.beta_linear
            *
            linear_pred
        )


        prediction = self.revin(
            pred_norm,
            "denorm"
        )

        return prediction