from __future__ import annotations

from typing import Optional

from ..metric import TrainingMetric
from ..training_state import TrainingState


class FullNoiseScaleMetric(TrainingMetric):
    """
    Full noise scale B_noise from https://arxiv.org/pdf/1812.06162 "An Empirical Model of Large-Batch Training"
    Eq 2.8: B_noise = tr(H Sigma) / G^T H G.

    L = (1/N) Σ_i l_i  full loss, N = full input set size
    g_i = ∇ l_i  per-sample gradient for all params θ∈R^D, D=Σ numel
    G = ∇L = (1/N) Σ_i g_i true gradient vector for all params over full set
    H = ∇²L = (1/N) Σ_i ∇² l_i true Hessian matrix for all params over full set, shape (D,D)
    Σ = Cov(g_i) = (1/N) Σ_i (g_i - G)(g_i - G)^T covariance over full set, shape (D,D)
    B_noise = tr(H Σ) / G^T H G
    """

    name = "b_noise"

    def __init__(
        self, loss_fn, model_key: str = "model", full_data_key: str = "full_data"
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.model_key = model_key
        self.full_data_key = full_data_key

    def measure(self, state: TrainingState) -> Optional[float]:
        model = state.get_extra(self.model_key)
        full_data = state.get_extra(self.full_data_key)
        if model is None or full_data is None:
            # BEFORE: model None or full_data None -> cannot compute true gradient/Hessian
            return None  # AFTER: None

        # BEFORE: need torch for autograd, model expected to have parameters method
        if not hasattr(model, "parameters"):
            # BEFORE: model does not have parameters, not nn.Module
            return None  # AFTER: None
        try:
            # BEFORE: attempt to import torch lazily to keep core no-torch-dep optional
            import torch  # AFTER: torch module available
        except ImportError:
            # BEFORE: torch not installed
            return None  # AFTER: None

        # full_data is tuple
        # x_full shape e.g, (5,2)=[[1,2],[3,4],[5,6],[7,8],[9,10]]
        # y_full shape e.g, (5,1)=[[3],[7],[11],[15],[19]]), N=5 is full input set size, D=3 is params
        x_full, y_full = full_data

        N = x_full.shape[0]

        # BEFORE: e.g, N=5 will be used for loops, D will be derived from model params
        D = sum(
            p.numel() for p in model.parameters()
        )  # AFTER: e.g, D=3 = Σ numel, numel means number of elements, e.g., weight (1,2) has 2 elements, bias (1,) has 1 element

        # Will hold e.g, N=5 vectors where each vector shape (3,) is g_i = gradient of per-sample loss l_i with respect to all parameters θ in R^D, D=3
        per_sample_grads = []
        # Will hold e.g, N=5 scalars
        per_sample_losses = []
        # Will hold e.g, N=5 matrices each shape (3,3)=(D=3,D=3)=∇² l_i
        per_sample_hessians = []

        for i in range(N):
            # BEFORE: x_full[i] e.g, shape (2,)=[1.,2.] from x_full shape (5,2)
            x_i = x_full[i].unsqueeze(
                0
            )  # AFTER: x_i e.g, shape (1,2)=[[1.,2.]] since unsqueeze adds dim 0 so (2,) becomes (1,2) as model expects (B=1,in=2)

            # BEFORE: y_full[i] e.g, shape (1,)=[3.] from y_full shape (5,1)
            y_i = y_full[i].unsqueeze(
                0
            )  # AFTER: y_i e.g, shape (1,1)=[[3.]] since unsqueeze adds dim 0 so (1,) becomes (1,1)

            # BEFORE: x_i e.g, shape (1,2)=[[1,2]]
            pred_i = model(x_i)  # AFTER: pred_i e.g, shape (1,1)=[[0.0]] for i=0

            # BEFORE: pred_i e.g, shape (1,1)=[[0.0]], y_i e.g, shape (1,1)=[[3.]]
            loss_i = self.loss_fn(
                pred_i, y_i
            )  # AFTER: loss_i e.g, shape (), a scala value 4.5 = l_i

            per_sample_losses.append(loss_i)

            # This line computes first-order gradient tuple where each element is gradient of l_i with respect to one parameter tensor
            # Tuple length equals number of parameter tensors in model e.g, 2 in this example
            # Why create_graph=True? It retains computation graph of first-order grads so we can differentiate again for second-order H_i
            # Why retain_graph=True? We call autograd.grad N times for first-order in this loop, without retain_graph graph freed after first call
            first_order_grad_tuple = torch.autograd.grad(
                loss_i, model.parameters(), create_graph=True, retain_graph=True
            )  # AFTER: e.g, first_order_grad_tuple is tuple of length 2 = number of param tensors, first_order_grad_tuple[0] is grad_weight shape (1,2) = [[-3.,-6.]] = ∂l_i/∂weight, first_order_grad_tuple[1] is grad_bias shape (1,) = [-3.] = ∂l_i/∂bias

            grad_vector = torch.cat(
                [g.flatten() for g in first_order_grad_tuple]
            )  # AFTER: grad_vector = ∇l_i e.g, [-3.,-6.,-3.] shape (3,)

            per_sample_grads.append(
                grad_vector
            )  # AFTER: per_sample_grads list of element shape (D,) e.g, = (3,) = [1,2,1]

            # H_i = ∇² l_i = second derivative matrix of per-sample loss l_i with respect to all params θ ∈ R^D e.g, shape (D,D)=(3,3) per sample
            H_i_rows = []

            # D is number of parameters
            for d in range(D):
                # BEFORE: grad_vector[d] scalar shape () e.g., -6. = ∂l_i/∂θ_d where d = 1
                second_order_grad_tuple = torch.autograd.grad(
                    grad_vector[d], model.parameters(), retain_graph=True
                )
                # AFTER e.g, second_order_grad_tuple is tuple of length 2  = number of param tensors, e.g., element0 shape (1,2) = [[1.,2.]] = ∂²l_i/∂w, element1 shape (1,) = [1.] = ∂²l_i/∂b,

                hessian_row = torch.cat(
                    [g.flatten() for g in second_order_grad_tuple]
                )  # AFTER: e.g, hessian_row is shape (3,)=[1.,2.,1.]

                H_i_rows.append(
                    hessian_row
                )  # AFTER: e.g, H_i_rows shape (D,D) e.g,[[1,2,1],[2,4,2],[1,2,1]] for i=0 example where x=[1,2]

            H_i = torch.stack(H_i_rows)

            per_sample_hessians.append(
                H_i
            )  # AFTER: e.g, per_sample_hessians is a list of elements shape (3,3)=(D=3,D=3)

        grad_matrix = torch.stack(
            per_sample_grads
        )  # AFTER: grad_matrix has a shape (N,d) e.g, (N=5,D=3) [[-3,-6,-3],[-5,-10,-5],[-7,-14,-7],[-9,-18,-9],[-11,-22,-11]] where row i is g_i = ∇l_i

        # G is true gradient vector for all params over full set: G = ∇L = (1/N) Σ_i g_i shape (D,) e.g, =(3,)
        # BEFORE: grad_matrix shape (N,D) e.g, (N=5,D=3) = [[-3,-6,-3],...], N=5 full set
        G = grad_matrix.mean(dim=0)  # AFTER: e.g G shape (3,)=[-7.,-14.,-7.]

        # centered_grads is g_i - G
        # BEFORE:  grad_matrix shape (N,D), e.g, (N=5,D=3) = [[-3,-6,-3],...], G shape (D,) e.g, (3,) = [-7,-14,-7] broadcast to (5,3)
        centered_grads = (
            grad_matrix - G
        )  # AFTER: centered_grads shape (N,D) e.g, (N=5,D=3) = [[4.,8.,4.],[2.,4.,2.],[0.,0.,0.],[-2.,-4.,-2.],[-4.,-8.,-4.]]

        # H is true Hessian over full set: H = ∇²L = (1/N) Σ_i ∇² l_i = mean_i H_i
        # BEFORE: per_sample_hessians shape (N,D,D) e.g, list of N=5 matrices [H_0,...,H_4] each shape (3,3), H_0=[[1,2,1],[2,4,2],[1,2,1]], etc.
        H = torch.stack(per_sample_hessians).mean(dim=0)
        # AFTER: H shape (D,D) e.g,(D=3,D=3) = [[11.,18.,3],[18,26,4],[3,4,1]]

        # Sigma = covariance over full set: Σ = (1/N) Σ_i (g_i - G)(g_i - G)^T shape (D,D) e.g, =(3,3)
        # centered_grads.unsqueeze(2) adds dimension 2: (N,D) -> (N,D,1) e.g, = (5,3,1). centered_grads (5,3)=[[4,8,4],...] becomes [[[4],[8],[4]],...] shape (5,3,1), each inner is column vector (D,1)=(3,1)=[[4],[8],[4]]
        # centered_grads.unsqueeze(1) adds dimension 1: (N,D) -> (N,1,D) e.g, = (5,1,3). centered_grads (5,3)=[[4,8,4],...] becomes [[[4,8,4]],...] shape (5,1,3), each inner is row vector (1,D)=(1,3)=[[4,8,4]]
        # Batched @ does matrix multiplication for each matrix along the first dimension N: (N,D,1) @ (N,1,D) -> (N,D,D) e.g, = (5,3,3). For each N, (D,1) @ (1,D) = (D,D), [[16,32,16],[32,64,32],[16,32,16]] shape (3,3)=(D,D)
        # mean(dim=0) aggregates on first dimension N: (N,D,D)=(5,3,3) -> (D,D)=(3,3) = (1/N) Σ_i outer_i
        Sigma = (centered_grads.unsqueeze(2) @ centered_grads.unsqueeze(1)).mean(
            dim=0
        )  # AFTER: Sigma shape (N, D) e.g, (D=3,D=3) = (1/N) Σ_i diff_i * diff_i^T

        # BEFORE: True Hessian H=∇²L shape (D, D) e.g, (D=3,D=3), true covariance Sigma shape (D, D) e.g, (D=3,D=3)
        H_Sigma = (
            H @ Sigma
        )  # AFTER: H_Sigma shape (D, D) e.g, (D=3,D=3) = H multiplied by Sigma

        # tr(H Σ) = trace = sum of diagonal Σ_d (H Σ)_{d,d}
        trace_H_Sigma = torch.trace(
            H_Sigma
        )  # AFTER: trace_H_Sigma scalar shape () = Σ_d (H Sigma)_{d,d} = tr(H Σ)

        # BEFORE: H shape (D, D), G true gradient vector ∇L for all params shape (D, ) e.g, (D=3,)=[-7.,-14.,-7.],
        H_G = H @ G  # AFTER: H_G shape (D,)

        # G^T H G = Σ_{d,e} G_d * H_{d,e} * G_e which is scala shape
        # G * H_G is element-wise multiplication, not matrix multiplication: (D,)*(D,) -> (D,) = [G_d * (H G)_d] shape (3,)
        # sum() aggregates D=3 dimensions Σ_d G_d*(H G)_d to scalar shape () because dot product G^T(H G)=Σ_d G_d*(H G)_d = Σ_{d,e} G_d*H_{d,e}*G_e
        G_H_G = (
            G * H_G
        ).sum()  # AFTER: G_H_G scalar shape () = Σ_d G_d*(H G)_d = G^T H G

        # BEFORE: trace_H_Sigma scalar () = tr(H Σ), G_H_G scalar () = G^T H G
        B_noise = (
            trace_H_Sigma / G_H_G
        )  # AFTER: B_noise scalar shape () = tr(H Σ) / G^T H G Eq2.8 true noise scale over full input set N

        if not torch.isfinite(B_noise):
            # BEFORE: B_noise may be non-finite if G_H_G zero
            return None  # AFTER: None
        return float(B_noise.item())  # AFTER: float B_noise e.g., 0.10
