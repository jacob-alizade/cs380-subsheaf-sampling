from abc import ABC, abstractmethod
import torch
import torch_sparse

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# from time import time

class SubsheafSampler(ABC):
    """
    Abstract base class for subsheaf samplers.
    """

    def __init__(self, num_nodes: int, d: int, k: int, device=None):
        self.num_nodes = num_nodes
        self.d = d
        self.k = k
        self.device = device if device is not None else torch.device("cpu")

    @abstractmethod
    def sample(self):
        """
        Return a tensor P of shape (num_nodes, d, k).
        """
        pass

class UniformSampler(SubsheafSampler):
    """
    Sampler that selects k distinct coordinate axes per node uniformly at random.
    """

    def __init__(self, num_nodes: int, d: int, k: int, device=None):
        super().__init__(num_nodes, d, k, device)
        self.I = torch.eye(d, device=self.device)

    @torch.no_grad()
    def sample(self) -> torch.Tensor:
        # sample k axes without replacement
        idx = torch.randperm(self.d, device=self.device)[:self.k]
        # build a single P frame: (d, k)
        P_frame = self.I[idx].t()
        # expand to all nodes: (num_nodes, d, k)
        return P_frame.unsqueeze(0).expand(self.num_nodes, -1, -1).clone()
    
class UniformSamplerSparse(SubsheafSampler):
    """
    Sampler that selects k distinct coordinate axes per node uniformly at random,
    but returns a sparse (num_nodes × d × k) tensor.
    """
    def __init__(self, num_nodes: int, d: int, k: int, device=None):
        super().__init__(num_nodes, d, k, device)

    @torch.no_grad()
    def sample(self) -> torch.Tensor:
        # Step 1: pick k distinct axis indices in [0..d)
        idx = torch.randperm(self.d, device=self.device)[:self.k]
        
        # Step 2: build the batched indices for a sparse (num_nodes, d, k) tensor
        # - batch_idx: for each node, repeat its index k times
        batch_idx = torch.arange(self.num_nodes, device=self.device).repeat_interleave(self.k)
        # - row_idx: for each node, the same idx array
        row_idx   = idx.repeat(self.num_nodes)
        # - col_idx: for each node, 0..k-1
        col_idx   = torch.arange(self.k, device=self.device).repeat(self.num_nodes)
        
        # Stack into a 3×(num_nodes*k) index tensor
        indices = torch.stack([batch_idx, row_idx, col_idx], dim=0)
        
        # All values are 1.0
        values = torch.ones(self.num_nodes * self.k, device=self.device)
        
        # Define the full sparse tensor shape
        shape = (self.num_nodes, self.d, self.k)
        
        P = torch.sparse_coo_tensor(indices, values, shape, device=self.device)
        # coalesce to canonicalize indices (optional but recommended)
        return P.coalesce()
    
# class RandomSampler(SubsheafSampler):
#     """
#     Sampler that returns a random orthonormal frame per node via batched QR on Gaussian.
#     """

#     @torch.no_grad()
#     def sample(self) -> torch.Tensor:
#         # 1) sample all Gaussians at once: (N, d, k)
#         A = torch.randn((self.num_nodes, self.d, self.k), device=self.device)
#         # 2) batched QR → Q: (N, d, k), R: (N, k, k)
#         #Q, R = torch.linalg.qr(A, mode='reduced')
#         # 3) fix sign‐ambiguity per column in each batch
#         #    diag(R) has shape (N, k)
#         #signs = torch.sign(torch.diagonal(R, offset=0, dim1=-2, dim2=-1))
#         #    broadcast to (N, d, k)
#         #Q = Q * signs.unsqueeze(1)
#         return Q
    
# class SpectralFrameSampler(SubsheafSampler):
#     """
#     Sampler that will use the precomputed d smallest nonzero eigenmodes
#     of the full sheaf Laplacian.
#     Now accepts a sparse Laplacian (edge_index, weights).
#     """

#     def __init__(self, num_nodes: int, d: int, k: int, device=None):
#         super().__init__(num_nodes, d, k, device)
#         self.L_full = None
#         self.evals = None
#         self.evecs = None
#         self.prev_X = None
#         self.Rs = None  # Store R factors from QR for future use

#     @torch.no_grad
#     def set_L(self, Lap: tuple):
#         """
#         Set and preprocess the full sheaf Laplacian from sparse form, with thorough checks.
#         Lap = (edge_index, weights)
#         """
#         edge_index, weights = Lap
#         Nd = self.num_nodes * self.d
        
#         # Build dense Laplacian
#         L_sparse = torch.sparse_coo_tensor(
#             edge_index, weights.view(-1),
#             (Nd, Nd),
#             device=self.device
#         )
#         L = L_sparse.to_dense()

#         # Eigen-decompose and cache the d smallest nonzero modes
#         vals, vecs = torch.lobpcg(
#             A=L,
#             B=None,
#             k=self.d,
#             X=self.prev_X,
#             niter=5,
#             tol=1e-6,
#             largest=False
#         )

#         self.prev_X = vecs

#         nz = torch.where(vals > 1e-12)[0]
#         idx = nz[torch.argsort(vals[nz])[:self.d]]

#         sorted_all = torch.argsort(vals)                          # shape (Nd,)s:
#         nullity = (vals.abs() < 1e-12).sum().item()

#         start = nullity

#         needed = self.d
#         end = start + needed
#         if end <= sorted_all.numel():
#             idx = sorted_all[start:end]
#         else:
#             # pad by recycling the smallest nonzero modes
#             extra = end - sorted_all.numel()
#             idx = torch.cat([sorted_all[start:], sorted_all[start:start+extra]], 0)

#         frames_raw = vecs[:, idx].view(self.num_nodes, self.d, self.d)
#         # Batched QR: orthonormalize each frame, store R's
#         Q, R = torch.linalg.qr(frames_raw)
#         self.evals = vals[idx]
#         self.evecs = Q
#         self.Rs = R

#         # torch.set_printoptions(sci_mode=False)
#         # print(self.evecs)

#     def sample(self) -> torch.Tensor:
#         """
#         Uniformly sample k eigenmode-columns (same indices for all nodes)
#         from the stored d frames in self.evecs.
#         Returns:
#         P: Tensor of shape (num_nodes, d, k)
#         """
#         # Pick k indices out of the d stored modes
#         idx = torch.randperm(self.d, device=self.device)[:self.k]
#         # Gather those columns for every node
#         P = self.evecs[:, :, idx]  # shape: (num_nodes, d, k)
#         return P



    

