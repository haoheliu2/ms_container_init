# flake8: noqa

import logging
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import torch
import torch.nn.functional as F

from numba import cuda, jit
from torch.autograd import Function
# from torchtts.trainers.transformer_trainer import TransformerHook

# plt.switch_backend('agg')
SAVE_ALIGNMENT_FLAG = False
GLOBAL_STEPS = 1
AVERAGE_OPTIMAL_PATH_MATRIX_DIR = r'./'
CROSS_ATTENTION_MATRIX_DIR = r'./'
CROSS_ATTENTION = None
logger = logging.getLogger(__name__)

def _plot_and_save(mat, image_optimal_path, figsize, dpi):
    plt.figure(figsize=figsize)
    plt.imshow(mat)
    plt.savefig(image_optimal_path,dpi=dpi)
    plt.close()

def manhattan(x, y):
    """ Computes Manhattan distance, summed over all dimensions, averaged over the batch.

    x: [b, t_x, c]
    y: [b, t_y, c]
    """
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    return F.l1_loss(x, y, reduction="none").mean(-1)


def smooth_l1(x, y):
    """ Computes smooth L1 distance, summed over all dimensions, averaged over the batch.

    x: [b, t_x, c]
    y: [b, t_y, c]
    """
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    return F.smooth_l1_loss(x, y, reduction="none").mean(-1)


def euclidean(x, y):
    """ Computes Euclidean distance, summed over all dimensions, averaged over the batch.

    x: [b, t_x, c]
    y: [b, t_y, c]
    """
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    return ((x - y) ** 2).mean(-1)



@cuda.jit
def compute_softdtw_cuda(D, penalty, gamma, bandwidth, max_i, max_j, n_passes, R, x_lens, y_lens):
    """ Computes cuda implementation of forward soft dynamic time warping loss. """
    # Each block processes one pair of examples.
    b = cuda.blockIdx.x

    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid.
    I_tid = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal.
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds.
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J).
        i = I_tid + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds.
        if I_tid + J == p and (I_tid < max_i and J < max_j):

            # Don't compute if outside bandwidth
            if i <= x_lens[b] and j <= y_lens[b]:
                if not (bandwidth < abs(j - i)):
                    r0 = -R[b, i - 1, j - 1] * inv_gamma
                    r1 = -(R[b, i - 1, j] + penalty) * inv_gamma
                    r2 = -(R[b, i, j - 1] + penalty) * inv_gamma
                    rmax = max(max(r0, r1), r2)
                    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                    softmin = -gamma * (math.log(rsum) + rmax)
                    R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block.
        cuda.syncthreads()


@cuda.jit
def compute_softdtw_backward_cuda(D, R, penalty, inv_gamma, bandwidth, max_i, max_j, n_passes, E, x_lens, y_lens):
    """ Computes cuda implementation of backward soft dynamic time warping loss. """
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to progress backwards.
    I_tid = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward.
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j.
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I_tid + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds.
        if I_tid + J == rev_p and (I_tid < max_i and J < max_j):

            # Don't compute if outside bandwidth
            if i <= x_lens[k] and j <= y_lens[k]:
                if math.isinf(R[k, i, j]):
                    R[k, i, j] = -math.inf
                if not (bandwidth < abs(j - i)):
                    a = math.exp((R[k, i + 1, j] - R[k, i, j] - penalty - D[k, i + 1, j]) * inv_gamma)
                    b = math.exp((R[k, i, j + 1] - R[k, i, j] - penalty - D[k, i, j + 1]) * inv_gamma)
                    c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                    E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block.
        cuda.syncthreads()


class _SoftDTWCUDA(Function):
    """ CUDA Soft DTW implementation.
    """

    @staticmethod
    def forward(ctx, D, penalty, gamma, bandwidth, mask, x_lens, y_lens):
        dev = D.device
        dtype = D.dtype
        penalty = torch.cuda.FloatTensor([penalty])
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])
        mask = mask.to(dev).type(dtype)  # float type
        D *= mask

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size
        # (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence
        # (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()), penalty.item(), gamma.item(), bandwidth.item(), N, M, n_passes, cuda.as_cuda_array(R), cuda.as_cuda_array(x_lens.detach()), cuda.as_cuda_array(y_lens.detach()))

        ctx.save_for_backward(D, R, penalty, gamma, bandwidth, mask, x_lens, y_lens)

        arange = torch.arange(B, device=dev)
        return R[arange, x_lens, y_lens]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        # print("output",dev)
        dtype = grad_output.dtype
        D, R, penalty, gamma, bandwidth, mask, x_lens, y_lens = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D
        R_ = R.detach()
        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)

        for n in range(B):
            E[n, x_lens[n] + 1, y_lens[n] + 1] = 1
            R_[n, x_lens[n] + 1:, :] = -math.inf
            R_[n, :, y_lens[n] + 1:] = -math.inf
            R_[n, x_lens[n] + 1, y_lens[n] + 1] = R_[n, x_lens[n], y_lens[n]]

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R_), penalty.item(),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E),
                                                            cuda.as_cuda_array(x_lens.detach()),
                                                            cuda.as_cuda_array(y_lens.detach()))

        E = E[:, 1:N + 1, 1:M + 1]
        E_mask = E * mask
        try:
            if SAVE_ALIGNMENT_FLAG:
                dir_path = AVERAGE_OPTIMAL_PATH_MATRIX_DIR
                image_optimal_path = os.path.join(dir_path, "Steps_{}_Average_Optimal_Path.jpg".format(GLOBAL_STEPS))
                dir_path = CROSS_ATTENTION_MATRIX_DIR
                image_cross_attention = os.path.join(dir_path, "Steps_{}_Cross_Attention.jpg".format(GLOBAL_STEPS))
                # import ipdb; ipdb.set_trace()
                os.makedirs(AVERAGE_OPTIMAL_PATH_MATRIX_DIR,exist_ok=True)
                os.makedirs(CROSS_ATTENTION_MATRIX_DIR,exist_ok=True)
                optimal_matrix = E_mask[-1].float().data.cpu().numpy()
                _plot_and_save(optimal_matrix, image_optimal_path, figsize=(8, 8), dpi=100)
                # attention_matrix = torch.stack([attn_cross[-1].transpose(1, 2) for attn_cross in CROSS_ATTENTION], dim=0)
                # TransformerHook._plot_and_save(attention_matrix.float().data.cpu().numpy(),
                #                                image_cross_attention,
                #                                figsize=(16, 4),
                #                                dpi=150)
                logger.info(f"Saving Soft-DTW Average-Optimal-Path picture to the {os.path.abspath(image_optimal_path)}")
                logger.info(f"Saving Cross-Attention picture to the {os.path.abspath(image_cross_attention)}")
        except:
            print("Error drawing attension map.")
        return grad_output.view(-1, 1, 1).expand_as(E) * E_mask, None, None, None, None, None, None


@jit(nopython=True)
def compute_softdtw(D, penalty, gamma, bandwidth, x_lens, y_lens):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in range(B):
        for j in range(1, M + 1):
            if j > y_lens[b]:
                continue
            for i in range(1, N + 1):
                if i > x_lens[b]:
                    continue
                # Check the pruning condition
                if bandwidth < abs(j - i):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -(R[b, i - 1, j] + penalty) / gamma
                r2 = -(R[b, i, j - 1] + penalty) / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@jit(nopython=True)
def compute_softdtw_backward(D_, R, penalty, gamma, bandwidth, x_lens, y_lens):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    assert B == x_lens.shape[0]
    for n in range(B):
        E[n, x_lens[n] + 1, y_lens[n] + 1] = 1
        R[n, x_lens[n] + 1:, :] = -np.inf
        R[n, :, y_lens[n] + 1:] = -np.inf
        R[n, x_lens[n] + 1, y_lens[n] + 1] = R[n, x_lens[n], y_lens[n]]

    for k in range(B):
        for j in range(M, 0, -1):
            if j > y_lens[k]:
                continue
            for i in range(N, 0, -1):
                if i > x_lens[k]:
                    continue

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf
                # Check the pruning condition
                if bandwidth < np.abs(j - i):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - penalty - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - penalty - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]


class _SoftDTW(Function):
    """ Vanilla soft DTW implementation.
    """

    @staticmethod
    def forward(ctx, D, penalty, gamma, bandwidth, mask, x_lens, y_lens):
        dev = D.device
        dtype = D.dtype
        penalty = torch.Tensor([penalty]).to(dev).type(dtype)
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        mask = mask.to(dev).type(dtype)
        D *= mask
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        p_ = penalty.item()

        x_lens_numpy = x_lens.detach().cpu().numpy()
        y_lens_numpy = y_lens.detach().cpu().numpy()

        R = torch.Tensor(compute_softdtw(D_, p_, g_, b_, x_lens_numpy, y_lens_numpy)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, penalty, gamma, bandwidth, mask, x_lens, y_lens)

        E = torch.Tensor(compute_softdtw_backward(D_, R.detach().cpu().numpy(), p_, g_, b_, x_lens_numpy, y_lens_numpy)).to(dev).type(dtype)
        E_mask = E * mask
        try:
            if SAVE_ALIGNMENT_FLAG:
                dir_path = AVERAGE_OPTIMAL_PATH_MATRIX_DIR
                image_optimal_path = os.path.join(dir_path, "Steps_{}_Average_Optimal_Path.jpg".format(GLOBAL_STEPS))
                dir_path = CROSS_ATTENTION_MATRIX_DIR
                image_cross_attention = os.path.join(dir_path, "Steps_{}_Cross_Attention.jpg".format(GLOBAL_STEPS))
                # import ipdb; ipdb.set_trace()
                os.makedirs(AVERAGE_OPTIMAL_PATH_MATRIX_DIR,exist_ok=True)
                os.makedirs(CROSS_ATTENTION_MATRIX_DIR,exist_ok=True)
                optimal_matrix = E_mask[-1].float().data.cpu().numpy()
                _plot_and_save(optimal_matrix, image_optimal_path, figsize=(8, 8), dpi=100)
                # attention_matrix = torch.stack([attn_cross[-1].transpose(1, 2) for attn_cross in CROSS_ATTENTION], dim=0)
                # TransformerHook._plot_and_save(attention_matrix.float().data.cpu().numpy(),
                #                                image_cross_attention,
                #                                figsize=(16, 4),
                #                                dpi=150)
                logger.info(f"Saving Soft-DTW Average-Optimal-Path picture to the {os.path.abspath(image_optimal_path)}")
                logger.info(f"Saving Cross-Attention picture to the {os.path.abspath(image_cross_attention)}")
        except:
            print("Error drawing attension map.")
        return R[torch.arange(x_lens.numel()), x_lens, y_lens], E_mask

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, penalty, gamma, bandwidth, mask, x_lens, y_lens = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        p_ = penalty.item()
        g_ = gamma.item()
        b_ = bandwidth.item()

        x_lens_numpy = x_lens.detach().cpu().numpy()
        y_lens_numpy = y_lens.detach().cpu().numpy()

        E = torch.Tensor(compute_softdtw_backward(D_, R_, p_, g_, b_, x_lens_numpy, y_lens_numpy)).to(dev).type(dtype)
        E_mask = E * mask
        try:
            if SAVE_ALIGNMENT_FLAG:
                dir_path = AVERAGE_OPTIMAL_PATH_MATRIX_DIR
                image_optimal_path = os.path.join(dir_path, "Steps_{}_Average_Optimal_Path.jpg".format(GLOBAL_STEPS))
                dir_path = CROSS_ATTENTION_MATRIX_DIR
                image_cross_attention = os.path.join(dir_path, "Steps_{}_Cross_Attention.jpg".format(GLOBAL_STEPS))
                # import ipdb; ipdb.set_trace()
                os.makedirs(AVERAGE_OPTIMAL_PATH_MATRIX_DIR,exist_ok=True)
                os.makedirs(CROSS_ATTENTION_MATRIX_DIR,exist_ok=True)
                optimal_matrix = E_mask[-1].float().data.cpu().numpy()
                _plot_and_save(optimal_matrix, image_optimal_path, figsize=(8, 8), dpi=100)
                # attention_matrix = torch.stack([attn_cross[-1].transpose(1, 2) for attn_cross in CROSS_ATTENTION], dim=0)
                # TransformerHook._plot_and_save(attention_matrix.float().data.cpu().numpy(),
                #                                image_cross_attention,
                #                                figsize=(16, 4),
                #                                dpi=150)
                logger.info(f"Saving Soft-DTW Average-Optimal-Path picture to the {os.path.abspath(image_optimal_path)}")
                logger.info(f"Saving Cross-Attention picture to the {os.path.abspath(image_cross_attention)}")
        except:
            print("Error drawing attension map.")
        return grad_output.view(-1, 1, 1).expand_as(E) * E_mask, None, None, None, None, None, None


class SoftDynamicTimeWarping(torch.nn.Module):
    """ High-level wrapper for soft dynamic time warmping implementation that optionally supports CUDA.

    Args:
        use_cuda: whether to use CUDA implementation
        gamma: soft minimum temperature parameter
        bandwidth: Sakoe-Chiba bandwidth for pruning, pass None to disable
        dist_func: optional point-wise distance function to use, defaults to smooth Manhattan distance
    """

    def __init__(
        self,
        penalty: float = 1.0,
        gamma: float = 0.01,
        bandwidth: int = None,
        dist_func: str = "manhattan",
        average_alignment_save_path=None,
        device='cuda'
    ):
        super().__init__()
        self.penalty = penalty
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = torch.cuda.is_available() and 'cuda' in device
        global AVERAGE_OPTIMAL_PATH_MATRIX_DIR
        AVERAGE_OPTIMAL_PATH_MATRIX_DIR = average_alignment_save_path + os.sep + 'average_optimal_path_matrix'
        global CROSS_ATTENTION_MATRIX_DIR
        CROSS_ATTENTION_MATRIX_DIR = average_alignment_save_path + os.sep + 'cross_attention_matrix'
        # Set the distance function
        self.dist_func = eval(dist_func)

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
            use_cuda = False
        # Finally, return the correct function
        return _SoftDTWCUDA.apply if use_cuda else _SoftDTW.apply

    def forward(self, x, y, x_mask, y_mask, save_alignment_flag=False, global_steps=1, cross_attention=None):
        """
        Compute the soft-DTW value between X and Y
        :param x: One batch of examples, batch_size x seq_len x dims
        :param y: The other batch of examples, batch_size x seq_len x dims
        :param x_mask: B x N (bool type), --> predict
        :param y_mask: B x M (bool type), --> groundtruth
        :param save_alignment_flag: False (bool type)
        :param global_steps: 1 (int type)
        :return: The computed results
        """
        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(x, y)
        global SAVE_ALIGNMENT_FLAG
        SAVE_ALIGNMENT_FLAG = save_alignment_flag and (global_steps % 1000 == 0)
        global GLOBAL_STEPS
        GLOBAL_STEPS = global_steps
        global CROSS_ATTENTION
        CROSS_ATTENTION = cross_attention

        n = x_mask.size(1)
        m = y_mask.size(1)
        x_mask_unsqueeze = x_mask.unsqueeze(2).expand(-1, n, m)
        y_mask_unsqueeze = y_mask.unsqueeze(1).expand(-1, n, m)
        mask = x_mask_unsqueeze & y_mask_unsqueeze  # bool type
        mask_cumsum = mask.cumsum(-1).cumsum(-2)
        x_lens = mask_cumsum[:, -1, 0].long()
        y_lens = mask_cumsum[:, 0, -1].long()

        # print('dist_funct')
        d_xy = self.dist_func(x, y)
        # print("apply")
        # print("==>", torch.max(torch.abs(d_xy)), torch.sum(torch.isnan(d_xy)).item(),torch.sum(torch.isinf(d_xy)).item())
        # if(torch.sum(torch.isnan(d_xy)).item() != 0 or torch.sum(torch.isinf(d_xy)).item() != 0):
        #     import ipdb; ipdb.set_trace()
        normalize_length = int(d_xy.size(-1) * d_xy.size(-2)) 
        result, E = func_dtw(d_xy, self.penalty, self.gamma, self.bandwidth, mask, x_lens, y_lens)
        result = result.mean()
        # print(result,torch.sum(d_xy<0), d_xy)
        # print("---->",result)
        # if(torch.sum(torch.isinf(result)) > 0):
        #     import ipdb; ipdb.set_trace()
        # result = func_dtw(d_xy, self.penalty, self.gamma, self.bandwidth, mask, x_lens, y_lens).mean()
        return (result / normalize_length) * 10000, E

if __name__ == '__main__':
    torch.manual_seed(0)
    def test_manhattan():
        soft_dtw = SoftDynamicTimeWarping(penalty=1.0,gamma=0.01,bandwidth=120,dist_func="manhattan",average_alignment_save_path=".",device="cpu")
        a = torch.randn((3, 10, 192))
        b = torch.randn((3, 20, 192))
        b[:,::2,:] = a.clone()
        # linear = torch.nn.Linear(192,192)
        # a = linear(a)
        a_mask = torch.ones((3,10)).bool()
        b_mask = torch.ones((3,20)).bool()
        
        # man = manhattan(a,b)
        # print(man,man.size())
        res = soft_dtw(a,b,a_mask,b_mask, save_alignment_flag=True,global_steps=1000)
        print(res)
        # res.backward()
        
    # test_manhattan()
    soft_dtw = SoftDynamicTimeWarping(penalty=0.1,gamma=0.01,bandwidth=120,dist_func="manhattan",average_alignment_save_path=".",device="cpu")
    
    shape1 = (1,1000,192)
    shape2 = (1,1000,192)
    
    z_p = torch.randn(shape1)
    logs_q = torch.randn(shape1)
    m_p = z_p # torch.randn(shape2)
    logs_p = logs_q # torch.randn(shape2)
    
    # linear = torch.nn.Linear(192,192)
    # z_p = linear(z_p)
    
    # logs_p[:,::2,:] = logs_q
    # m_p[:,::2,:] = z_p
    a_mask = torch.ones((1,1000)).bool()
    b_mask = torch.ones((1,1000)).bool()
    x = torch.cat([z_p, logs_q], dim=-1)
    y = torch.cat([m_p, logs_p], dim=-1)
    print(torch.sum(x-y))
    res = soft_dtw(x,y,a_mask,b_mask,save_alignment_flag=True, global_steps=1000)
    print(res)
    # res.backward()
    
    # print(torch.mean(torch.abs(m-n)))