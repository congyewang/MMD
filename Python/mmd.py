import torch

import jax.numpy as jnp
from jax import jit

from sklearn.metrics.pairwise import pairwise_kernels

from tqdm.auto import trange


def compute_mmd(x, y, kernel='rbf'):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y.
    
    Parameters:
        - x: First set of samples. Shape: (n, d)
        - y: Second set of samples. Shape: (m, d)
        - kernel: Type of kernel to be used. Currently supports 'rbf'.
        - bandwidth: Bandwidth for the RBF kernel.
        
    Returns:
        - mmd: The MMD value between x and y.
    """

    def rbf_kernel(x, y, bandwidth=1.0):
        x = x.unsqueeze(1)  # Shape: (n, 1, d)
        y = y.unsqueeze(0)  # Shape: (1, m, d)
        
        # Compute pairwise distances
        distances = (x - y).pow(2).sum(-1)

        return torch.exp(-distances / (2 * bandwidth * bandwidth))

    def imq_kernel(x, y, c=1.0):
        x = x.unsqueeze(1)  # Shape: (n, 1, d)
        y = y.unsqueeze(0)  # Shape: (1, m, d)

        # Compute pairwise distances
        distances = (x - y).pow(2).sum(-1)
        
        return c * c / (c * c + distances)

    def linear_kernel(x, y):
        return torch.mm(x, y.t())

    def cosine_similarity_kernel(x, y):
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (y.norm(p=2, dim=1, keepdim=True) + 1e-8)
        return torch.mm(x_norm, y_norm.t())

    if kernel == 'rbf':
        kernel_func = rbf_kernel
    elif kernel == 'imq':
        kernel_func = imq_kernel
    elif kernel == 'linear':
        kernel_func = linear_kernel
    elif kernel == 'cosine':
        kernel_func = cosine_similarity_kernel
    else:
        raise ValueError("Unsupported kernel type.")

    with torch.no_grad():
        # Compute individual kernel matrices
        xx_kernel = kernel_func(x, x).mean()
        yy_kernel = kernel_func(y, y).mean()
        xy_kernel = kernel_func(x, y).mean()

    # Compute MMD
    mmd = xx_kernel + yy_kernel - 2 * xy_kernel
    return mmd

def compute_mmd_final_efficient(x, y, kernel='rbf', batch_size=500):
    """
    Final memory-efficient computation of MMD.
    
    Parameters:
        - x: First set of samples. Shape: (n, d)
        - y: Second set of samples. Shape: (m, d)
        - kernel: Type of kernel to be used. Currently supports 'rbf'.
        - batch_size: Size of batch to compute kernel values.
        
    Returns:
        - mmd: The MMD value between x and y.
    """
    n, _ = x.shape
    m = y.shape[0]

    def rbf_kernel(x, y, bandwidth=1.0):
        # Compute pairwise distances
        distances = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
        return torch.exp(-distances / (2 * bandwidth * bandwidth))

    if kernel == 'rbf':
        kernel_func = rbf_kernel
    else:
        raise ValueError("Unsupported kernel type.")

    xx_sum, yy_sum, xy_sum = 0.0, 0.0, 0.0
    count_xx, count_yy, count_xy = 0, 0, 0

    with torch.no_grad():  # Ensure that no gradients are computed
        for i in trange(0, n, batch_size):
            x_batch = x[i:min(i+batch_size, n)]
            
            xx_kernel_batch = kernel_func(x_batch, x_batch)
            xx_sum += xx_kernel_batch.sum().item()
            count_xx += x_batch.shape[0] * x_batch.shape[0]
            
            for j in range(0, m, batch_size):
                y_batch = y[j:min(j+batch_size, m)]
                
                xy_kernel_batch = kernel_func(x_batch, y_batch)
                xy_sum += xy_kernel_batch.sum().item()
                count_xy += x_batch.shape[0] * y_batch.shape[0]
                
                if i == j:
                    yy_kernel_batch = kernel_func(y_batch, y_batch)
                    yy_sum += yy_kernel_batch.sum().item()
                    count_yy += y_batch.shape[0] * y_batch.shape[0]
                
                del y_batch, xy_kernel_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            del x_batch, xx_kernel_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    xx_kernel = xx_sum / count_xx
    yy_kernel = yy_sum / count_yy
    xy_kernel = xy_sum / count_xy

    # Compute MMD
    mmd = xx_kernel + yy_kernel - 2 * xy_kernel

    return mmd

def compute_mmd_sklearn(x, y, kernel='rbf'):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y using sklearn.
    
    Parameters:
        - x: First set of samples. Shape: (n, d)
        - y: Second set of samples. Shape: (m, d)
        - gamma: Parameter for the RBF kernel in sklearn.
        
    Returns:
        - mmd: The MMD value between x and y.
    """
    # Convert PyTorch tensors to numpy arrays
    x = x.numpy()
    y = y.numpy()

    # Compute individual kernel matrices
    xx_kernel = pairwise_kernels(x, x, metric=kernel).mean()
    yy_kernel = pairwise_kernels(y, y, metric=kernel).mean()
    xy_kernel = pairwise_kernels(x, y, metric=kernel).mean()

    # Compute MMD
    mmd = xx_kernel + yy_kernel - 2 * xy_kernel
    return mmd

def compute_mmd_batched(x, y, kernel='rbf', batch_size=100):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y, using batched computation.
    
    Parameters:
        - x: First set of samples. Shape: (n, d)
        - y: Second set of samples. Shape: (m, d)
        - kernel: Type of kernel to be used. Currently supports 'rbf'.
        - batch_size: The size of batches for computing the kernel matrices.
        
    Returns:
        - mmd: The MMD value between x and y.
    """
    
    def rbf_kernel(x, y, bandwidth=1.0):
        x = x.unsqueeze(1)  # Shape: (n, 1, d)
        y = y.unsqueeze(0)  # Shape: (1, m, d)
        
        # Compute pairwise distances
        distances = (x - y).pow(2).sum(-1)

        return torch.exp(-distances / (2 * bandwidth * bandwidth))

    def imq_kernel(x, y, c=1.0):
        x = x.unsqueeze(1)  # Shape: (n, 1, d)
        y = y.unsqueeze(0)  # Shape: (1, m, d)

        # Compute pairwise distances
        distances = (x - y).pow(2).sum(-1)
        
        return c * c / (c * c + distances)

    def linear_kernel(x, y):
        return torch.mm(x, y.t())

    def cosine_similarity_kernel(x, y):
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (y.norm(p=2, dim=1, keepdim=True) + 1e-8)
        return torch.mm(x_norm, y_norm.t())

    if kernel == 'rbf':
        kernel_func = rbf_kernel
    elif kernel == 'imq':
        kernel_func = imq_kernel
    elif kernel == 'linear':
        kernel_func = linear_kernel
    elif kernel == 'cosine':
        kernel_func = cosine_similarity_kernel
    else:
        raise ValueError("Unsupported kernel type.")

    with torch.no_grad():
        # Initialize accumulators for kernel values
        xx_sum, yy_sum, xy_sum, count = 0.0, 0.0, 0.0, 0

        # Compute kernel matrices in batches
        for i in trange(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            for j in range(0, len(y), batch_size):
                y_batch = y[j:j + batch_size]

                xx_sum += kernel_func(x_batch, x_batch).sum().item()
                yy_sum += kernel_func(y_batch, y_batch).sum().item()
                xy_sum += kernel_func(x_batch, y_batch).sum().item()
                
                count += x_batch.shape[0] * y_batch.shape[0]

        # Compute mean kernel values
        xx_kernel = xx_sum / count
        yy_kernel = yy_sum / count
        xy_kernel = xy_sum / count

    # Compute MMD
    mmd = xx_kernel + yy_kernel - 2 * xy_kernel
    return mmd

@jit
def rbf_kernel(x, y, bandwidth=1.0):
    # Compute pairwise distances
    distances = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-distances / (2 * bandwidth * bandwidth))

def compute_mmd_final_efficient_jax(x, y, kernel='rbf', batch_size=10_000):
    """
    Final memory-efficient computation of MMD using JAX.
    
    Parameters:
        - x: First set of samples. Shape: (n, d)
        - y: Second set of samples. Shape: (m, d)
        - kernel: Type of kernel to be used. Currently supports 'rbf'.
        - batch_size: Size of batch to compute kernel values.
        
    Returns:
        - mmd: The MMD value between x and y.
    """
    n, _ = x.shape
    m = y.shape[0]

    def rbf_kernel(x, y, bandwidth=1.0):
        # Compute pairwise distances
        distances = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        return jnp.exp(-distances / (2 * bandwidth * bandwidth))

    if kernel == 'rbf':
        kernel_func = rbf_kernel
    else:
        raise ValueError("Unsupported kernel type.")

    xx_sum, yy_sum, xy_sum = 0.0, 0.0, 0.0
    count_xx, count_yy, count_xy = 0.0, 0.0, 0.0  # Convert to float

    for i in trange(0, n, batch_size):
        x_batch = x[i:min(i+batch_size, n)]
        
        xx_kernel_batch = kernel_func(x_batch, x_batch)
        xx_sum += jnp.sum(xx_kernel_batch)
        count_xx += float(x_batch.shape[0] * x_batch.shape[0])
        
        for j in range(0, m, batch_size):
            y_batch = y[j:min(j+batch_size, m)]
            
            xy_kernel_batch = kernel_func(x_batch, y_batch)
            xy_sum += jnp.sum(xy_kernel_batch)
            count_xy += float(x_batch.shape[0] * y_batch.shape[0])
            
            if i == j:
                yy_kernel_batch = kernel_func(y_batch, y_batch)
                yy_sum += jnp.sum(yy_kernel_batch)
                count_yy += float(y_batch.shape[0] * y_batch.shape[0])

    xx_kernel = xx_sum / count_xx
    yy_kernel = yy_sum / count_yy
    xy_kernel = xy_sum / count_xy

    # Compute MMD
    mmd = xx_kernel + yy_kernel - 2 * xy_kernel

    return mmd


if __name__ == "__main__":
    x_torch = torch.randn(100_000, 2)
    y_torch = torch.randn(100_000, 2) + 2.0

    x_gpu = x_torch.to('cuda').half()
    y_gpu = y_torch.to('cuda').half()

    # del x_torch
    # del y_torch

    # mmd_value = compute_mmd(x_gpu, y_gpu, kernel='rbf')
    # mmd_value

    mmd_value_batched = compute_mmd_batched(x_gpu, y_gpu, kernel='rbf', batch_size=100)
    mmd_value_batched

    compute_mmd_final_efficient_compile = torch.compile(compute_mmd_final_efficient)
    mmd_value_opt = compute_mmd_final_efficient_compile(x_torch, y_torch, kernel='rbf', batch_size=10_000)
    mmd_value_opt

    compute_mmd_sklearn(x_torch, y_torch, kernel="rbf")

    for i in ['linear', 'rbf', 'cosine']:
        print(f"{i}\n{compute_mmd(x_torch, y_torch, kernel=i)}\t{compute_mmd_sklearn(x_torch, y_torch, kernel=i)}\n")

    x_np = x_torch.numpy()
    y_np = y_torch.numpy()

    x_jax = jnp.array(x_np)
    y_jax = jnp.array(y_np)

    compute_mmd_final_efficient_jax(x_jax, y_jax, kernel='rbf', batch_size=10_000)
