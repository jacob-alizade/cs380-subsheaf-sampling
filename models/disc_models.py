# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse

from torch import nn
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant
from models.samplers import UniformSampler, UniformSamplerSparse

from time import perf_counter


class DiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)
        
        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            #print(x.shape, self.graph_size, self.final_d)

            # Use the adjacency matrix rather than the diagonal
            #time_0 = perf_counter()
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            #print(perf_counter()-time_0)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    
def DiscreteSampleBundleSheafDiffusion(k):

    class _DiscreteSampleBundleSheafDiffusion(SheafDiffusion):

        def __init__(self, edge_index, args):
            """
            Subsheaf-sampling extension of SheafDiffusion

            Args:
                edge_index:           Graph connectivity tensor
                args:                 Base SheafDiffusion arguments dict
                k (int):              Target subsheaf dimension (frame size)
                sample_budget (int):  Number of (node,coord) samples to draw
                compute_P (bool):     If True, compute P via local PCA; else expect P passed externally
            """
            super().__init__(edge_index, args)
            self.k = k
            self.sample_budget = args["sample_budget"]

            self.final_k = self.k
            if self.add_hp:
                self.final_k += 1
            if self.add_lp:
                self.final_k += 1

            self.final_d = self.d
            self.hidden_dim = args['hidden_channels'] * self.d

            self.generator = torch.Generator(device=self.device)
            self.generator.seed()  # seed from OS entropy, not the global torch seed


            self.lin_right_weights = nn.ModuleList()
            self.lin_left_weights = nn.ModuleList()

            self.batch_norms = nn.ModuleList()
            if self.right_weights:
                for i in range(self.layers):
                    self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                    nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
            if self.left_weights:
                for i in range(self.layers):
                    self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                    nn.init.eye_(self.lin_left_weights[-1].weight.data)

            self.sheaf_learners = nn.ModuleList()
            self.weight_learners = nn.ModuleList()

            num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
            for i in range(num_sheaf_learners):
                if self.sparse_learner:
                    self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                        self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
                else:
                    self.sheaf_learners.append(LocalConcatSheafLearner(
                        self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
                
                if self.use_edge_weights:
                    self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))

            self.full_builder = lb.NormConnectionLaplacianBuilder(
                self.graph_size,
                edge_index,
                d=self.d,
                add_hp=False,
                add_lp=False,
                orth_map=self.orth_trans,
            )

            self.laplacian_builder = lb.SubsheafLaplacianBuilder(
                self.graph_size,
                edge_index,
                k=self.final_k,
                # add_hp=self.add_hp,
                # add_lp=self.add_lp,
            )
            
            self.epsilons = nn.ParameterList()
            for i in range(self.layers):
                self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

            self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
            if self.second_linear:
                self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        def project_right_weights(self, W: torch.Tensor) -> torch.Tensor:
            P = self.P  # (V, d, k)
            V = P.size(0)
            W_exp = W.unsqueeze(0).expand(V, self.final_d, self.final_d)  # (V, d, d)
            # W_k[v] = P[v].T @ W @ P[v]
            return torch.bmm(P.transpose(1, 2), torch.bmm(W_exp, P))  # (V, k, k)
        
        def get_param_size(self):
            if self.orth_trans in ['matrix_exp', 'cayley']:
                return self.d * (self.d + 1) // 2
            else:
                return self.d * (self.d - 1) // 2
        
        def left_right_linear(self, x, left, right):
            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = left(x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = right(x)

            return x
        
        def update_edge_index(self, edge_index):
            super().update_edge_index(edge_index)
            for weight_learner in self.weight_learners:
                weight_learner.update_edge_index(edge_index)
       
        def forward(self, x):
            # ——— Initial feature transforms ———
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.second_linear:
                x = self.lin12(x)

            # reshape into (N*d, feat)
            N, d, k = self.graph_size, self.final_d, self.final_k
            x = x.view(N * d, -1)
            x0 = x
            L_k = None

            # ——— Layer loop ———
            for layer in range(self.layers):
                 # sample k axes by index
                idx_base = torch.randperm(d, device=self.device, generator=self.generator)  # (k,)
                idx = idx_base[:self.k]
                idx2 = idx_base[:self.final_k]

                # restrict transports via slicing
                if layer == 0 or self.nonlinear:
                    x_maps = F.dropout(x, p=(self.dropout if layer > 0 else 0.0), training=self.training).view(N, -1)
                    map_params = self.sheaf_learners[layer](x_maps, self.edge_index)
                    maps_full = self.full_builder.orth_transform(map_params)

                    
                    # maps_full: (E, d, d) → (E, k, k)
                    # tmp = maps_full[:, idx, :]        # (E, k, d)
                    # maps_k = tmp[:, :, idx]           # (E, k, k)

                    E = maps_full.size(0)
                    maps_k_padded = torch.zeros(E, self.final_k, self.final_k, device=self.device)   # (E, d, d) of zeros

                    #print(f"{maps_k_padded.shape=} {maps_k.shape=}")
                    gap = self.final_k - self.k
                    # Insert the d–2 slice into rows/cols idx
                    maps_k_padded[:, :(-gap if gap else None), :][:, :, :(-gap if gap else None)] = maps_full[:, idx, :][:, :, idx]

                    assert gap in [0,1,2]

                    # Now tack on the LP (+1) and HP (–1) in their own slots:
                    if self.add_lp:
                        if gap == 1:
                            maps_k_padded[:, -2, -2] = 1
                        else:
                            maps_k_padded[:, -1, -1] = 1
                    if self.add_hp:
                        if gap == 2:
                            maps_k_padded[:, -2, -2] = -1
                        else:
                            maps_k_padded[:, -1, -1] = -1

                    #print(f"{maps_k_padded[0]}")
                    # edge weights if used
                    edge_weights = (
                        self.weight_learners[layer](x_maps, self.edge_index)
                        if self.use_edge_weights else None
                    )

                    
                    
                    L_k, trans_maps_k = self.laplacian_builder(
                        maps_k_padded, edge_weights
                    )

                    row, col = L_k[0]   # edge_index
                    max_idx = max(row.max().item(), col.max().item())
                    dim = max_idx + 1
                    #print(f"L_k is a {dim} × {dim} sparse matrix")
                    
                    self.sheaf_learners[layer].set_L(trans_maps_k)


                # left/right linear + dropout
                x_trans = F.dropout(x, p=self.dropout, training=self.training)
                #print(f"{x_trans.shape=}")
                x_trans = self.left_right_linear(
                    x_trans,
                    self.lin_left_weights[layer],
                    self.lin_right_weights[layer]
                )
                #print(f"{x_trans.shape=}")
                # reshape x_trans → (N, d, feat)
                x_full = x_trans.view(N, d, -1)
                #print(f"{x_full.shape=}")

                # project into subspace by slicing: (N, k, feat)
                x_sub = x_full[:, idx2, :]

                #print(f"{x_sub.shape=} {self.final_k=}")

                # flatten for diffusion
                # 1) Compute how many extra dims to add:
                # pad = self.final_k - self.k    # 4 - 2 = 2

                # # 2) Pad with zeros so that x_sub becomes (N, final_k, F) = (7, 4, 20)
                # if pad > 0:
                #     zeros = x_sub.new_ones(N, pad, self.hidden_channels)   # shape (7, 2, 20)
                #     #print(f"{zeros.shape=}")
                #     x_sub = torch.cat([x_sub, zeros], dim=1)
                #     # now x_sub.shape == (7, 4, 20)

                # 3) Now flatten correctly into (N*final_k, F) = (28, 20)
                x_sub_flat = x_sub.view(N * self.final_k, self.hidden_channels)
                x_sub_flat = torch_sparse.spmm(
                    L_k[0], L_k[1],
                    N *  self.final_k, N *  self.final_k,
                    x_sub_flat
                )



                # back to (N, k, feat)
                x_sub = x_sub_flat.view(N,  self.final_k, -1)

                #print(x_sub)

                #print(x_sub.shape, self.final_k)

                # lift back by scatter
                x_full2 = torch.zeros_like(x_full)


                x_full2[:, idx2, :] = x_sub

                # accumulate
                x_full2 = x_full2.view(N * d, -1)

                # activation on accumulator
                if self.use_act:
                    x_full2 = F.elu(x_full2)

                # residual update with proper broadcasting
                eps = torch.tanh(self.epsilons[layer]).view(1, d, 1)
                x0_reshaped = x0.view(N, d, -1)
                x0_reshaped = (1.0 + eps) * x0_reshaped - x_full2.view(N, d, -1)
                x0 = x0_reshaped.view(N * d, -1)
                x = x0

            # ——— Final output ———
            out = x0.view(self.graph_size, -1)
            out = self.lin2(out)
            return F.log_softmax(out, dim=1)
        
    return _DiscreteSampleBundleSheafDiffusion



class DiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)