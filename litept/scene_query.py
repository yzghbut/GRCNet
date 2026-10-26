# Scene tokens provide global context with shared low-rank projections.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geobias import geometry_reliability


class GlobalSceneQuery(nn.Module):


    def __init__(
        self,
        channels,
        num_tokens=8,
        query_rank=16,
        num_coord_frequencies=4,
        proj_drop=0.0,
        layer_scale_init=1.0e-3,
        num_classes=0,
        assignment_mode="softmax",
        sinkhorn_iterations=5,
        sinkhorn_epsilon=0.1,
        sinkhorn_mass_relaxation=0.5,
        use_geometry_reliability=False,
        geometry_k=(4, 6, 10),
        shared_basis=False,
        replace_ffn=False,
        ffn_rank=128,
        output_adapter_rank=0,
        local_reliability_floor=0.5,
        global_reliability_floor=0.5,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if query_rank <= 0:
            raise ValueError("query_rank must be positive")
        if num_coord_frequencies <= 0:
            raise ValueError("num_coord_frequencies must be positive")
        if int(num_classes) < 0:
            raise ValueError("num_classes must be non-negative")
        assignment_mode = str(assignment_mode).lower()
        if assignment_mode not in {"softmax", "sinkhorn", "unbalanced_sinkhorn"}:
            raise ValueError(
                "assignment_mode must be 'softmax', 'sinkhorn', "
                "or 'unbalanced_sinkhorn'"
            )
        if int(sinkhorn_iterations) <= 0:
            raise ValueError("sinkhorn_iterations must be positive")
        if float(sinkhorn_epsilon) <= 0.0:
            raise ValueError("sinkhorn_epsilon must be positive")
        if not 0.0 < float(sinkhorn_mass_relaxation) <= 1.0:
            raise ValueError("sinkhorn_mass_relaxation must be in (0, 1]")

        self.channels = int(channels)
        self.num_tokens = int(num_tokens)
        self.query_rank = int(query_rank)
        self.num_coord_frequencies = int(num_coord_frequencies)
        self.num_classes = int(num_classes)
        self.assignment_mode = assignment_mode
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_mass_relaxation = float(sinkhorn_mass_relaxation)
        self.use_geometry_reliability = bool(use_geometry_reliability)
        self.geometry_k = geometry_k
        self.shared_basis = bool(shared_basis)
        self.replace_ffn = bool(replace_ffn)
        self.ffn_rank = int(ffn_rank)
        self.output_adapter_rank = int(output_adapter_rank)
        self.local_reliability_floor = float(local_reliability_floor)
        self.global_reliability_floor = float(global_reliability_floor)
        if self.replace_ffn and self.ffn_rank <= 0:
            raise ValueError("ffn_rank must be positive when replace_ffn is enabled")
        if self.output_adapter_rank < 0:
            raise ValueError("output_adapter_rank must be non-negative")
        if not 0.0 <= self.local_reliability_floor <= 1.0:
            raise ValueError("local_reliability_floor must be in [0, 1]")
        if not 0.0 <= self.global_reliability_floor <= 1.0:
            raise ValueError("global_reliability_floor must be in [0, 1]")

        coord_channels = 6 * self.num_coord_frequencies
        self.norm = nn.LayerNorm(self.channels)
        self.token_assignment = nn.Linear(self.channels, self.num_tokens)
        self.token_mass_prior = (
            nn.Linear(self.channels, self.num_tokens)
            if self.assignment_mode == "unbalanced_sinkhorn"
            else None
        )
        if self.token_mass_prior is not None:


            nn.init.zeros_(self.token_mass_prior.weight)
            nn.init.zeros_(self.token_mass_prior.bias)
        if self.shared_basis:
            self.shared_latent = nn.Linear(self.channels, self.query_rank)
            self.token_value = nn.Identity()
            self.point_query = nn.Identity()
            self.output = nn.Identity()
            self.output_bias = nn.Parameter(torch.zeros(self.channels))
            self.token_value_scale = nn.Parameter(torch.ones(self.query_rank))
            self.point_query_scale = nn.Parameter(torch.ones(self.query_rank))
            self.output_scale = nn.Parameter(torch.ones(self.query_rank))
            if self.output_adapter_rank > 0:


                self.output_adapter_down = nn.Linear(
                    self.query_rank, self.output_adapter_rank, bias=False
                )
                self.output_adapter_up = nn.Linear(
                    self.output_adapter_rank, self.channels, bias=False
                )
                nn.init.zeros_(self.output_adapter_up.weight)
            else:
                self.output_adapter_down = None
                self.output_adapter_up = None
        else:
            self.shared_latent = None
            self.token_value = nn.Linear(self.channels, self.query_rank)
            self.point_query = nn.Linear(self.channels, self.query_rank)
            self.output = nn.Linear(self.query_rank, self.channels)
            self.register_parameter("output_bias", None)
            self.register_parameter("token_value_scale", None)
            self.register_parameter("point_query_scale", None)
            self.register_parameter("output_scale", None)
            if self.output_adapter_rank > 0:
                self.output_adapter_down = nn.Linear(
                    self.query_rank, self.output_adapter_rank, bias=False
                )
                self.output_adapter_up = nn.Linear(
                    self.output_adapter_rank, self.channels, bias=False
                )
                nn.init.zeros_(self.output_adapter_up.weight)
            else:
                self.output_adapter_down = None
                self.output_adapter_up = None
        self.coord_query = nn.Linear(coord_channels, self.query_rank, bias=False)
        self.token_key = nn.Linear(self.query_rank, self.query_rank)
        self.semantic_head = (
            nn.Linear(self.query_rank, self.num_classes)
            if self.num_classes > 0
            else None
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer_scale = nn.Parameter(
            torch.full((self.channels,), float(layer_scale_init))
        )
        if self.use_geometry_reliability:
            self.reliability_assignment = nn.Linear(
                1, self.num_tokens, bias=False
            )
            self.reliability_query = nn.Linear(1, self.query_rank, bias=False)
            nn.init.zeros_(self.reliability_assignment.weight)
            nn.init.zeros_(self.reliability_query.weight)
        else:
            self.reliability_assignment = None
            self.reliability_query = None
        if self.replace_ffn:
            self.ffn_down = nn.Linear(self.channels, self.ffn_rank)


            self.ffn_up = nn.Linear(self.ffn_rank, self.channels, bias=False)
            self.ffn_condition = nn.Linear(self.query_rank, self.ffn_rank)
            self.ffn_output_bias = nn.Parameter(torch.zeros(self.channels))
            self.ffn_layer_scale = nn.Parameter(
                torch.full((self.channels,), float(layer_scale_init))
            )
            nn.init.zeros_(self.ffn_condition.weight)
            nn.init.zeros_(self.ffn_condition.bias)
            with torch.no_grad():
                self.ffn_up.weight.copy_(self.ffn_down.weight.transpose(0, 1))
        else:
            self.ffn_down = None
            self.ffn_up = None
            self.ffn_condition = None
            self.register_parameter("ffn_output_bias", None)
            self.register_parameter("ffn_layer_scale", None)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):


        up_key = prefix + "ffn_up.weight"
        down_key = prefix + "ffn_down.weight"
        if self.replace_ffn and up_key not in state_dict and down_key in state_dict:
            state_dict[up_key] = state_dict[down_key].transpose(0, 1).clone()


        for name, module in (
            ("output_adapter_down.weight", self.output_adapter_down),
            ("output_adapter_up.weight", self.output_adapter_up),
        ):
            key = prefix + name
            if module is not None and key not in state_dict:
                state_dict[key] = module.weight.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _normalize_coord(coord):
        center = coord.mean(dim=0, keepdim=True)
        scale = (coord - center).square().mean().sqrt().clamp_min(1.0e-6)
        return (coord - center) / scale

    def _coord_features(self, coord):
        coord = self._normalize_coord(coord.float())
        frequencies = torch.pow(
            2.0,
            torch.arange(
                self.num_coord_frequencies,
                device=coord.device,
                dtype=coord.dtype,
            ),
        )
        phase = math.pi * coord[:, :, None] * frequencies[None, None, :]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(1)

    def _token_assignment(self, scores, row_mass=None):

        if self.assignment_mode == "softmax":
            return torch.softmax(scores, dim=-1)

        output_dtype = scores.dtype
        scores = scores.float()
        num_tokens, num_points = scores.shape
        log_assignment = scores / self.sinkhorn_epsilon
        log_col_mass = scores.new_full(
            (num_points,), math.log(float(num_tokens) / float(num_points))
        )
        if self.assignment_mode == "sinkhorn":


            for _ in range(self.sinkhorn_iterations):
                log_assignment = (
                    log_assignment
                    - torch.logsumexp(log_assignment, dim=1, keepdim=True)
                )
                log_assignment = (
                    log_assignment
                    - torch.logsumexp(log_assignment, dim=0, keepdim=True)
                    + log_col_mass[None, :]
                )
            return log_assignment.exp().to(dtype=output_dtype)


        if row_mass is None:
            row_mass = scores.new_full((num_tokens,), 1.0)
        row_mass = row_mass.float().clamp_min(torch.finfo(scores.dtype).tiny)
        row_mass = row_mass / row_mass.sum().clamp_min(torch.finfo(scores.dtype).tiny)
        row_mass = row_mass * float(num_tokens)
        log_row_mass = row_mass.log()
        log_u = scores.new_zeros(num_tokens)
        log_v = scores.new_zeros(num_points)
        relaxation = self.sinkhorn_mass_relaxation
        for _ in range(self.sinkhorn_iterations):
            log_u = relaxation * (
                log_row_mass
                - torch.logsumexp(log_assignment + log_v[None, :], dim=1)
            )
            log_v = log_col_mass - torch.logsumexp(
                log_assignment + log_u[:, None], dim=0
            )
        log_assignment = log_assignment + log_u[:, None] + log_v[None, :]
        return log_assignment.exp().to(dtype=output_dtype)

    def _feature_projection(self, normalized, role):
        if not self.shared_basis:
            return getattr(self, role)(normalized)
        latent = self.shared_latent(normalized)
        return latent * getattr(self, f"{role}_scale")[None, :]

    def _decode_output(self, response, global_gate=None):
        if not self.shared_basis:
            decoded = self.output(response)
        else:
            response = response * self.output_scale[None, :]
            decoded = F.linear(
                response,
                self.shared_latent.weight.transpose(0, 1),
                self.output_bias,
            )
        if self.output_adapter_up is not None:
            adapter = self.output_adapter_up(
                F.gelu(self.output_adapter_down(response))
            )
            if global_gate is not None:
                adapter = adapter * global_gate
            decoded = decoded + adapter
        return decoded

    def _conditioned_ffn(self, normalized, response):
        hidden = F.gelu(self.ffn_down(normalized))


        condition = 2.0 * torch.sigmoid(self.ffn_condition(response))
        hidden = hidden * condition
        return self.ffn_up(hidden) + self.ffn_output_bias

    def _decode_cloud(self, feat, coord, reliability=None):
        normalized = self.norm(feat)
        if reliability is None:
            reliability = normalized.new_ones((normalized.shape[0],))
        reliability = reliability.float().clamp(0.0, 1.0)


        assignment_scores = self.token_assignment(normalized).transpose(0, 1)
        if self.reliability_assignment is not None:
            assignment_scores = assignment_scores + self.reliability_assignment(
                reliability[:, None]
            ).transpose(0, 1).to(assignment_scores.dtype)
        if self.assignment_mode == "unbalanced_sinkhorn":
            mass_logits = self.token_mass_prior(normalized.mean(dim=0))
            row_mass = self.num_tokens * torch.softmax(
                mass_logits.float(), dim=-1
            )
        else:
            row_mass = None
        assignment = self._token_assignment(assignment_scores, row_mass=row_mass)
        scene_tokens = assignment @ self._feature_projection(normalized, "token_value")


        query = self._feature_projection(normalized, "point_query")
        query = query + self.coord_query(self._coord_features(coord)).to(query.dtype)
        if self.reliability_query is not None:
            query = query + self.reliability_query(reliability[:, None]).to(query.dtype)
        key = self.token_key(scene_tokens)
        attention_scores = query @ key.transpose(0, 1) / math.sqrt(self.query_rank)
        attention = torch.softmax(attention_scores, dim=-1)
        response = attention @ scene_tokens
        reliability_column = reliability.to(feat.dtype)[:, None]


        global_gate = self.global_reliability_floor + (
            1.0 - self.global_reliability_floor
        ) * reliability_column
        decoded = feat + self.layer_scale * global_gate * self.proj_drop(
            self._decode_output(response, global_gate=global_gate)
        )
        if self.replace_ffn:
            local_gate = 1.0 - (
                1.0 - self.local_reliability_floor
            ) * reliability_column
            decoded = decoded + self.ffn_layer_scale * local_gate * self.proj_drop(
                self._conditioned_ffn(normalized, response)
            )

        semantic_logits = None
        if self.semantic_head is not None:


            token_log_prob = F.log_softmax(
                self.semantic_head(scene_tokens).float(), dim=-1
            )
            point_log_prob = torch.logsumexp(
                torch.log(attention.float().clamp_min(torch.finfo(torch.float32).tiny))[
                    :, :, None
                ]
                + token_log_prob[None, :, :],
                dim=1,
            )
            semantic_logits = point_log_prob.to(feat.dtype)
        return decoded, semantic_logits

    def forward(self, point):
        feat = point.feat
        coord = point.coord
        batch = point.batch.long()
        output = torch.empty_like(feat)
        semantic_output = (
            feat.new_empty((feat.shape[0], self.num_classes))
            if self.semantic_head is not None
            else None
        )
        reliability = None
        if self.use_geometry_reliability:
            reliability = geometry_reliability(point, k=self.geometry_k).to(feat.dtype)
            if hasattr(point, "__setitem__"):
                point["geometry_reliability"] = reliability
            else:
                point.geometry_reliability = reliability


            reliability_float = reliability.float().clamp(0.0, 1.0)
            point.scene_query_reliability_mean = reliability_float.mean().detach()
            point.scene_query_reliability_min = reliability_float.min().detach()
            point.scene_query_reliability_max = reliability_float.max().detach()
            point.scene_query_global_gate_mean = (
                self.global_reliability_floor
                + (1.0 - self.global_reliability_floor) * reliability_float
            ).mean().detach()
            point.scene_query_local_gate_mean = (
                1.0
                - (1.0 - self.local_reliability_floor) * reliability_float
            ).mean().detach()

        for batch_id in torch.unique(batch, sorted=True):
            index = torch.nonzero(batch == batch_id, as_tuple=False).flatten()
            cloud_reliability = None if reliability is None else reliability[index]
            decoded, semantic_logits = self._decode_cloud(
                feat[index], coord[index], reliability=cloud_reliability
            )
            output[index] = decoded
            if semantic_output is not None:
                semantic_output[index] = semantic_logits

        point.feat = output
        if semantic_output is not None:
            point.scene_query_semantic_logits = semantic_output
        if "sparse_conv_feat" in point.keys():
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(output)
        return point
