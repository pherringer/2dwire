import numpy as np
import numba as nb

@nb.njit
def bsp_single_numba(a, b):
    """
    Binary symplectic product of two vectors. Dtype uint8. Indexed site-by-site, i.e. (x_1, z_1, x_2, z_2, ...).
    """
    assert not len(a) % 2
    n = len(a)//2
    result = np.uint8(0)
    for i in range(n):
        result += a[2*i]*b[2*i+1] + a[2*i+1]*b[2*i]
    return result % 2


def xor(a, b):
    """Element-wise XOR for tuples."""
    return tuple((a[i] + b[i]) % 2 for i in range(len(a)))


@nb.njit
def lambda_right(A):
    """Reproduces the action of the symplectic form acting on A
    from the right."""
    assert not A.shape[1] % 2
    result = np.zeros_like(A)
    result[:,::2] = A[:,1::2]
    result[:,1::2] = A[:,::2]
    return result


@nb.njit
def lambda_left(A):
    """Reproduces the action of the symplectic form acting on A
    from the left."""
    assert not A.shape[0] % 2 
    result = np.zeros_like(A)
    result[::2] = A[1::2]
    result[1::2] = A[::2]
    return result