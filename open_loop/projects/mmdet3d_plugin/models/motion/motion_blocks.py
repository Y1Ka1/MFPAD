import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
)

from projects.mmdet3d_plugin.core.box3d import *
from ..blocks import linear_relu_ln


class PlanRegMemoryForgetting(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=6,
        mem_hidden=256,
        trans_layers=2,
        trans_heads=4,
        inst_dim=128,
        dropout_forget=0.2,
        ins_k=32,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.mem_hidden = mem_hidden
        self.trans_layers = trans_layers
        self.trans_heads = trans_heads
        self.inst_dim = inst_dim
        self.dropout_forget = dropout_forget
        self.ins_k = ins_k

        self.init_h = nn.Linear(embed_dims, mem_hidden)
        self.init_c = nn.Linear(embed_dims, mem_hidden)

        self.lstm_input_dim = 2 + embed_dims
        self.lstm_cell = nn.LSTMCell(self.lstm_input_dim, mem_hidden)
        self.mem_out = nn.Linear(mem_hidden, 2)

        self.mem_proj = nn.Linear(mem_hidden, mem_hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=mem_hidden,
            nhead=trans_heads,
            dim_feedforward=mem_hidden * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=trans_layers)
        self.time_embed = nn.Parameter(torch.randn(self.fut_ts, mem_hidden))

        if inst_dim is not None:
            self.inst_kv_proj = nn.Linear(inst_dim, mem_hidden)
        else:
            self.inst_kv_proj = None

        self.corr_head = nn.Linear(mem_hidden, 2)
        self.gate_net = nn.Sequential(
            nn.Linear(mem_hidden + mem_hidden, mem_hidden),
            nn.ReLU(),
            nn.Linear(mem_hidden, 1),
        )

        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.xavier_uniform_(self.init_c.weight)
        nn.init.xavier_uniform_(self.mem_out.weight)
        nn.init.xavier_uniform_(self.mem_proj.weight)
        nn.init.xavier_uniform_(self.corr_head.weight)
        if self.inst_kv_proj is not None:
            nn.init.xavier_uniform_(self.inst_kv_proj.weight)

    def _build_q_global(self, plan_query, plan_weights=None):
        if plan_query.dim() == 2:
            return plan_query
        if plan_query.dim() >= 3:
            if plan_query.dim() == 4 and plan_query.size(1) == 1:
                tokens = plan_query.squeeze(1)
            else:
                tokens = plan_query
            if plan_weights is not None:
                weights = torch.softmax(plan_weights, dim=-1)
                return (weights.unsqueeze(-1) * tokens).sum(dim=1)
            return tokens.mean(dim=1)
        return plan_query

    def forward(self, plan_query, plan_weights=None, ins_feats=None, train_mode=True):
        device = plan_query.device
        q_global = self._build_q_global(plan_query, plan_weights=plan_weights)
        h = torch.tanh(self.init_h(q_global))
        c = torch.tanh(self.init_c(q_global))

        last_xy = torch.zeros(q_global.size(0), 2, device=device)
        mem_h_seq = []
        mem_pred_seq = []
        for _ in range(self.fut_ts):
            lstm_in = torch.cat([last_xy, q_global], dim=-1)
            h, c = self.lstm_cell(lstm_in, (h, c))
            h_do = self.dropout(h) if train_mode else h
            delta = self.mem_out(h_do)
            last_xy = last_xy + delta
            mem_h_seq.append(h.unsqueeze(0))
            mem_pred_seq.append(last_xy.unsqueeze(1))
        mem_h_seq = torch.cat(mem_h_seq, dim=0)
        mem_pred = torch.cat(mem_pred_seq, dim=1)

        memory = self.mem_proj(mem_h_seq)
        if ins_feats is not None and self.inst_kv_proj is not None:
            if ins_feats.dim() == 2:
                ins_feats = ins_feats.unsqueeze(1)
            if ins_feats.size(1) > self.ins_k:
                ins_feats = ins_feats[:, :self.ins_k, :]
            ins_proj = self.inst_kv_proj(ins_feats).permute(1, 0, 2)
            memory = torch.cat([memory, ins_proj], dim=0)

        tgt = self.time_embed.unsqueeze(1).repeat(1, q_global.size(0), 1).to(device)
        if train_mode and self.dropout_forget > 0:
            mem_len = memory.size(0)
            drop_mask = torch.rand(mem_len, memory.size(1), device=device) < self.dropout_forget
            memory = memory.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        decoded = self.transformer_decoder(tgt, memory).permute(1, 0, 2)
        decoded = self.dropout(decoded) if train_mode else decoded

        mem_proj_t = self.mem_proj(mem_h_seq.permute(1, 0, 2))
        gate_input = torch.cat([mem_proj_t, decoded], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input)).squeeze(-1)

        corr = self.corr_head(decoded)
        traj = mem_pred + gate.unsqueeze(-1) * corr
        return traj, gate


class PlanRegMemoryForgettingGRU(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=6,
        mem_hidden=256,
        trans_layers=2,
        trans_heads=4,
        inst_dim=128,
        dropout_forget=0.2,
        ins_k=32,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.mem_hidden = mem_hidden
        self.trans_layers = trans_layers
        self.trans_heads = trans_heads
        self.inst_dim = inst_dim
        self.dropout_forget = dropout_forget
        self.ins_k = ins_k

        self.init_h = nn.Linear(embed_dims, mem_hidden)

        self.rnn_input_dim = 2 + embed_dims
        self.gru_cell = nn.GRUCell(self.rnn_input_dim, mem_hidden)
        self.mem_out = nn.Linear(mem_hidden, 2)

        self.mem_proj = nn.Linear(mem_hidden, mem_hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=mem_hidden,
            nhead=trans_heads,
            dim_feedforward=mem_hidden * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=trans_layers)
        self.time_embed = nn.Parameter(torch.randn(self.fut_ts, mem_hidden))

        if inst_dim is not None:
            self.inst_kv_proj = nn.Linear(inst_dim, mem_hidden)
        else:
            self.inst_kv_proj = None

        self.corr_head = nn.Linear(mem_hidden, 2)
        self.gate_net = nn.Sequential(
            nn.Linear(mem_hidden + mem_hidden, mem_hidden),
            nn.ReLU(),
            nn.Linear(mem_hidden, 1),
        )

        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.xavier_uniform_(self.mem_out.weight)
        nn.init.xavier_uniform_(self.mem_proj.weight)
        nn.init.xavier_uniform_(self.corr_head.weight)
        if self.inst_kv_proj is not None:
            nn.init.xavier_uniform_(self.inst_kv_proj.weight)

    def _build_q_global(self, plan_query, plan_weights=None):
        if plan_query.dim() == 2:
            return plan_query
        if plan_query.dim() >= 3:
            if plan_query.dim() == 4 and plan_query.size(1) == 1:
                tokens = plan_query.squeeze(1)
            else:
                tokens = plan_query
            if plan_weights is not None:
                weights = torch.softmax(plan_weights, dim=-1)
                return (weights.unsqueeze(-1) * tokens).sum(dim=1)
            return tokens.mean(dim=1)
        return plan_query

    def forward(self, plan_query, plan_weights=None, ins_feats=None, train_mode=True):
        device = plan_query.device
        q_global = self._build_q_global(plan_query, plan_weights=plan_weights)
        h = torch.tanh(self.init_h(q_global))

        last_xy = torch.zeros(q_global.size(0), 2, device=device)
        mem_h_seq = []
        mem_pred_seq = []
        for _ in range(self.fut_ts):
            rnn_in = torch.cat([last_xy, q_global], dim=-1)
            h = self.gru_cell(rnn_in, h)
            h_do = self.dropout(h) if train_mode else h
            delta = self.mem_out(h_do)
            last_xy = last_xy + delta
            mem_h_seq.append(h.unsqueeze(0))
            mem_pred_seq.append(last_xy.unsqueeze(1))
        mem_h_seq = torch.cat(mem_h_seq, dim=0)
        mem_pred = torch.cat(mem_pred_seq, dim=1)

        memory = self.mem_proj(mem_h_seq)
        if ins_feats is not None and self.inst_kv_proj is not None:
            if ins_feats.dim() == 2:
                ins_feats = ins_feats.unsqueeze(1)
            if ins_feats.size(1) > self.ins_k:
                ins_feats = ins_feats[:, :self.ins_k, :]
            ins_proj = self.inst_kv_proj(ins_feats).permute(1, 0, 2)
            memory = torch.cat([memory, ins_proj], dim=0)

        tgt = self.time_embed.unsqueeze(1).repeat(1, q_global.size(0), 1).to(device)
        if train_mode and self.dropout_forget > 0:
            mem_len = memory.size(0)
            drop_mask = torch.rand(mem_len, memory.size(1), device=device) < self.dropout_forget
            memory = memory.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        decoded = self.transformer_decoder(tgt, memory).permute(1, 0, 2)
        decoded = self.dropout(decoded) if train_mode else decoded

        mem_proj_t = self.mem_proj(mem_h_seq.permute(1, 0, 2))
        gate_input = torch.cat([mem_proj_t, decoded], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input)).squeeze(-1)

        corr = self.corr_head(decoded)
        traj = mem_pred + gate.unsqueeze(-1) * corr
        return traj, gate


class PlanRegTransformerOnly(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=6,
        trans_layers=2,
        trans_heads=4,
        inst_dim=128,
        dropout_forget=0.2,
        ins_k=32,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.trans_layers = trans_layers
        self.trans_heads = trans_heads
        self.inst_dim = inst_dim
        self.dropout_forget = dropout_forget
        self.ins_k = ins_k

        self.token_proj = nn.Linear(embed_dims, embed_dims)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=trans_heads,
            dim_feedforward=embed_dims * 4,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=trans_layers)
        self.time_embed = nn.Parameter(torch.randn(self.fut_ts, embed_dims))

        if inst_dim is not None:
            self.inst_kv_proj = nn.Linear(inst_dim, embed_dims)
        else:
            self.inst_kv_proj = None

        self.corr_head = nn.Linear(embed_dims, 2)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 1),
        )

        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.xavier_uniform_(self.corr_head.weight)
        if self.inst_kv_proj is not None:
            nn.init.xavier_uniform_(self.inst_kv_proj.weight)

    def _flatten_tokens(self, plan_query, plan_weights=None):
        if plan_query.dim() == 2:
            tokens = plan_query.unsqueeze(1)
        elif plan_query.dim() == 4 and plan_query.size(1) == 1:
            tokens = plan_query.squeeze(1)
        else:
            tokens = plan_query
        if plan_weights is not None and tokens.dim() == 3:
            weights = torch.softmax(plan_weights, dim=-1)
            tokens = tokens * weights.unsqueeze(-1)
        return tokens

    def forward(self, plan_query, plan_weights=None, ins_feats=None, train_mode=True):
        device = plan_query.device
        tokens = self._flatten_tokens(plan_query, plan_weights=plan_weights)
        memory = self.token_proj(tokens).permute(1, 0, 2)

        if ins_feats is not None and self.inst_kv_proj is not None:
            if ins_feats.dim() == 2:
                ins_feats = ins_feats.unsqueeze(1)
            if ins_feats.size(1) > self.ins_k:
                ins_feats = ins_feats[:, :self.ins_k, :]
            ins_proj = self.inst_kv_proj(ins_feats).permute(1, 0, 2)
            memory = torch.cat([memory, ins_proj], dim=0)

        tgt = self.time_embed.unsqueeze(1).repeat(1, tokens.size(0), 1).to(device)
        if train_mode and self.dropout_forget > 0:
            mem_len = memory.size(0)
            drop_mask = torch.rand(mem_len, memory.size(1), device=device) < self.dropout_forget
            memory = memory.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        decoded = self.transformer_decoder(tgt, memory).permute(1, 0, 2)
        decoded = self.dropout(decoded) if train_mode else decoded

        corr = self.corr_head(decoded)
        gate = torch.sigmoid(self.gate_net(decoded)).squeeze(-1)
        traj = gate.unsqueeze(-1) * corr
        return traj, gate

@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        use_memory=True,
        use_forgetting=True,
        inst_dim=256,
        dropout_forget=0.2,
        ins_k=32,
    ):
        super(MotionPlanningRefinementModule, self).__init__()
        #import pdb;pdb.set_trace()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_memory = use_memory
        self.use_forgetting = use_forgetting

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        if use_memory:
            self.plan_reg_branch = PlanRegMemoryForgetting(
                embed_dims=embed_dims,
                fut_ts=ego_fut_ts,
                mem_hidden=embed_dims,
                trans_layers=2,
                trans_heads=4,
                inst_dim=inst_dim,
                dropout_forget=dropout_forget if use_forgetting else 0.0,
                ins_k=ins_k,
            )
        else:
            self.plan_reg_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, ego_fut_ts * 2),
            )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def _pool_plan_query(self, plan_query, plan_weights=None):
        if plan_query.dim() == 2:
            return plan_query
        if plan_query.dim() >= 3:
            if plan_query.dim() == 4 and plan_query.size(1) == 1:
                tokens = plan_query.squeeze(1)
            else:
                tokens = plan_query
            if plan_weights is not None:
                weights = torch.softmax(plan_weights, dim=-1)
                return (weights.unsqueeze(-1) * tokens).sum(dim=1)
            return tokens.mean(dim=1)
        return plan_query

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
        ins_feats=None,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_weights = plan_cls.squeeze(1) if plan_cls.dim() == 3 else plan_cls
        if self.use_memory:
            plan_traj, _gate = self.plan_reg_branch(
                plan_query,
                plan_weights=plan_weights,
                ins_feats=ins_feats,
                train_mode=self.training,
            )
        else:
            q_global = self._pool_plan_query(plan_query, plan_weights=plan_weights)
            plan_reg_flat = self.plan_reg_branch(q_global)
            plan_traj = plan_reg_flat.reshape(bs, self.ego_fut_ts, 2)
        plan_reg = plan_traj.unsqueeze(1).unsqueeze(1).repeat(1, 1, 3 * self.ego_fut_mode, 1, 1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModuleGRU(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        inst_dim=256,
        dropout_forget=0.2,
        ins_k=32,
    ):
        super(MotionPlanningRefinementModuleGRU, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = PlanRegMemoryForgettingGRU(
            embed_dims=embed_dims,
            fut_ts=ego_fut_ts,
            mem_hidden=embed_dims,
            trans_layers=2,
            trans_heads=4,
            inst_dim=inst_dim,
            dropout_forget=dropout_forget,
            ins_k=ins_k,
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
        ins_feats=None,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_weights = plan_cls.squeeze(1) if plan_cls.dim() == 3 else plan_cls
        plan_traj, _gate = self.plan_reg_branch(
            plan_query,
            plan_weights=plan_weights,
            ins_feats=ins_feats,
            train_mode=self.training,
        )
        plan_reg = plan_traj.unsqueeze(1).unsqueeze(1).repeat(1, 1, 3 * self.ego_fut_mode, 1, 1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModuleTransformerOnly(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        inst_dim=256,
        dropout_forget=0.2,
        ins_k=32,
    ):
        super(MotionPlanningRefinementModuleTransformerOnly, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.plan_reg_branch = PlanRegTransformerOnly(
            embed_dims=embed_dims,
            fut_ts=ego_fut_ts,
            trans_layers=2,
            trans_heads=4,
            inst_dim=inst_dim,
            dropout_forget=dropout_forget,
            ins_k=ins_k,
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
        ins_feats=None,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_weights = plan_cls.squeeze(1) if plan_cls.dim() == 3 else plan_cls
        plan_traj, _gate = self.plan_reg_branch(
            plan_query,
            plan_weights=plan_weights,
            ins_feats=ins_feats,
            train_mode=self.training,
        )
        plan_reg = plan_traj.unsqueeze(1).unsqueeze(1).repeat(1, 1, 3 * self.ego_fut_mode, 1, 1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status


class MotionPlanning2thRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=6,
        ego_fut_mode=3,
    ):
        super(MotionPlanning2thRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.plan_cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        plan_query,
        ego_feature,
        ego_anchor_embed,
    ):
        bs = plan_query.shape[0]
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return plan_cls, plan_reg, planning_status
