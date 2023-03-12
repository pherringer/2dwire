import numpy as np
import numba as nb


@nb.njit
def bsp_single_numba(a, b):
    """
    Binary symplectic product of two vectors. Dtype uint8.
    Indexed site-by-site, i.e. (x_1, z_1, x_2, z_2, ...).
    
    Tested faster than numpy for a.size <= 1e6.
    """
    assert not len(a) % 2
    n = len(a)//2
    result = np.uint8(0)
    for i in range(n):
        result += a[2*i]*b[2*i+1] + a[2*i+1]*b[2*i]
    return result % 2


def bsp_single_numpy(A, B):
    """Binary symplectic product of two vectors.
    
    Tested slower than numba for a.size <= 1e6.
    """
    A1 = A[::2]
    A2 = A[1::2]
    B1 = B[::2]
    B2 = B[1::2]
    return (A1@B2 + A2@B1) % 2


@nb.njit
def bsp_numba(A, B):
    """Binary symplectic product of two arrays.
    
    Tested faster than numpy for A.shape = (10, 1000).
    Tested slower than numpy for A.shape = (100, 1000).
    """
    result = np.zeros((A.shape[0], B.shape[0]), dtype=np.uint8)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            result[i, j] = bsp_single_numba(A[i], B[j])
    return result


def bsp_numpy(A, B):
    """
    Binary symplectic product of two arrays.
    
    Tested slower than numba for A.shape = (10, 1000).
    Tested faster than numba for A.shape = (100, 1000).
    """
    A1 = A[:,::2]
    A2 = A[:,1::2]
    B1 = B[:,::2]
    B2 = B[:,1::2]
    return np.hstack((A1, A2))@np.vstack((B2.T, B1.T)) % 2


def to_tuple(arr):
    """Convert 2D array to tuple of tuples."""
    return tuple(map(tuple, arr))


def xor(a, b):
    """Element-wise XOR for tuples."""
    return tuple((a[i] + b[i]) % 2 for i in range(len(a)))


def symplectic_gs(S):
    """Performs the symplectic Gram-Schmidt algorithm to give a symplectic
    basis for the a set of vectors S as desribed in arXiv:1406.2170.
    """
    basis = []
    while len(S):
        v = S[0]
        for w in S:
            if bsp_single_numba(v, w):
                basis.append(v)
                basis.append(w)
                S.remove(v)
                S.remove(w)
                for i in range(len(S)):
                    if bsp_single_numba(S[i], v) == 1:
                        S[i] = xor(S[i], w)
                    if bsp_single_numba(S[i], w) == 1:
                        S[i] = xor(S[i], v)
                break
        else:
            S.remove(v)
    return basis


def symplectic_ext_tuple(S):
    """Extends the set of binary vectors S to a symplectic basis.
    S is a list of tuples."""
    E = [tuple(x) for x in np.eye(len(S[0]), dtype=np.uint8)]
    return symplectic_gs(S + E)


def symplectic_ext_arr(S):
    """Extends the set of binary vectors S to a symplectic basis.
    S is a 2D binary array."""
    return np.array(
        symplectic_ext_tuple([tuple(x) for x in S]), dtype=S.dtype)


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


def lambda_mat(dim):
    assert not dim % 2
    return np.kron(
        np.eye(dim//2, dtype=np.uint8), 
        np.array([[0, 1], [1, 0]], dtype=np.uint8))


def is_symplectic(A):
    assert A.shape[0] == A.shape[1]
    assert not A.shape[0] % 2
    L = lambda_mat(A.shape[0])
    return np.array_equal(L, (A.T@lambda_left(A)) % 2)


def symplectic_inverse(A):
    """Returns the inverse of a binary symplectic matrix A"""
    assert is_symplectic(A)
    return lambda_left(lambda_right(A.T))


def symplectic_transf(A1, A2):
    """Constructs a binary symplectic matrix U such that 
    A2 = U@A1 for A1, A2 binary symplectic matrices."""
    assert is_symplectic(A1)
    assert is_symplectic(A2)
    return A2@symplectic_inverse(A1) % 2
