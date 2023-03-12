import numpy as np
import numba as nb

@nb.njit
def swap_rows(A, i, j):
    """Swap row i and j of a matrix A. A will be modified.

    Args:
        A (array, 2d): matrix
        i (int): first row to be swapped
        j (int): second row to be swapped
    """
    temp = A[i, :].copy()
    A[i, :] = A[j, :].copy()
    A[j, :] = temp


@nb.njit
def row_elim(A, i, j):
    """Eliminate leading entry in column j of all rows except i.

    A will be modified. Does not check to make sure that A is
    a binary matrix because the check takes too long. Passing a matrix
    that is not binary may result in logical errors.

    Args:
        A (array, 2d): matrix
        i (int): row with a leading 1
        j (_type_): column with a leading 1
    """
    assert A[i,j] == 1, 'No leading 1 at i, j'
    # Index of rows j < i with a 1 in the current column
    pre_idx = np.nonzero(A[:i,j])[0]
    # Index of rows j > i with a 1 in the current column
    post_idx = np.nonzero(A[i+1:,j])[0] + (i+1)
    # Index of all rows j \neq i with a 1 in the current column
    idx = np.concatenate((pre_idx, post_idx))
    # Eliminate all other 1's in the current column
    A[idx,:] = (A[idx,:] + A[i]) % 2


@nb.njit
def rref(A, mincol=0, maxcol=0, in_place=False):
    """Return the RREF form of a matrix A. By default A is not
    modified.

    Args:
        A (array, 2d): matrix
        mincol (int, optional): Column at which to start
            the row reudction procedure. Defaults to 0.
        maxcol (int, optional): Column at which to end the
            row reduction procedure. Defaults to 0.
        in_place (bool, optional): If True, A will be modified. 
            Defaults to False.

    Returns:
        array, 2d: RREF form of the matrix A.
    """
    # Avoids modifying the input matrix
    if not in_place:
        A = A.copy()
    nrow, ncol = A.shape
    # Row and column indices
    i = 0
    j = 0
    # If (mincol, maxcol) is given, 
    # only row reduce inside that range                           
    if mincol:
        j = mincol 
    if maxcol:
        ncol = min(maxcol, ncol)
    while i < nrow and j < ncol:
        # Find first nonzero entry in this column, if any
        if A[i:, j].any():
            row_id = np.argmax(A[i:, j]) + i
            swap_rows(A, i, row_id)
            row_elim(A, i, j)
            # Move to next row and column
            i += 1
            j += 1
        else:
            # If this column is all zeros, move to next column
            j += 1
    if not in_place:
        return A


@nb.njit
def rank(A, in_place=False):
    """Return the matrix rank of A.

    Args:
        A (array, 2d): matrix
        in_place (bool, optional): If True, A will be modified. 
            Defaults to False.

    Returns:
        int: the matrix rank of A.
    """
    if in_place:
        rref(A, in_place=True)
        return np.count_nonzero(np.sum(A, axis=1))
    else:
        R = rref(A, in_place=False)
        return np.count_nonzero(np.sum(R, axis=1))


@nb.njit
def nullity(A, in_place=False):
    """Return the dimension of the kernel of A.

    Args:
        A (array, 2d): matrix
        in_place (bool, optional): If True, A will be modified. 
            Defaults to False.

    Returns:
        int: the dimension of the kernel of A.
    """
    return A.shape[1] - rank(A, in_place=in_place)


@nb.njit
def reduce_and_trim(A, in_place=False):
    """Row reduce the matrix A and return only the nonzero rows.

    Args:
        A (array, 2d): matrix
        in_place (bool, optional): If True, A will be modified. 
            Defaults to False.

    Returns:
        array, 2d: RREF form of the matrix A, with all zero rows
            removed.
    """
    if not in_place:
        A = A.copy()
    r = rank(A, in_place=True)
    return A[:r]


@nb.njit
def kernel(A):
    """Return a basis for the kernel of A.

    Args:
        A (array, 2d): matrix

    Returns:
        array, 2d: Array whose rows are a basis for the 
            kernel of A.
    """
    A = A.copy()
    rref(A, in_place=True)
    E = np.eye(A.shape[1], dtype=A.dtype)
    B = np.concatenate((A.T, E), axis=1)
    rref(B, in_place=True)
    r = np.count_nonzero(np.sum(B[:, :A.shape[0]], axis=1))
    return B[r:, A.shape[0]:]


@nb.njit
def image(A):
    """Return a basis for the image of A.

    Args:
        A (array, 2d): matrix

    Returns:
        array, 2d: Array whose rows are a basis for the 
            image of A.
    """
    A = A.copy()
    rref(A, in_place=True)
    E = np.eye(A.shape[1], dtype=A.dtype)
    B = np.concatenate((A.T, E), axis=1)
    rref(B, in_place=True)
    r = np.count_nonzero(np.sum(B[:, :A.shape[0]], axis=1))
    return B[:r, A.shape[0]:]


@nb.njit
def inverse(A):
    """Find the matrix inverse of A."""
    assert A.shape[0] == A.shape[1], 'A is not square'
    assert rank(A) == A.shape[0], 'A is not invertible'
    n = A.shape[0]
    A_aug = np.hstack((A, np.eye(n, dtype=A.dtype)))
    rref(A_aug, in_place=True)
    return A_aug[:, n:]


@nb.njit
def is_in_span_nb(V, W):
    """Return True if V is in the span of W"""
    # if V.ndim == 1:
    #     V = V.reshape(1, -1)
    # if W.ndim == 1:
    #     W = W.reshape(1, -1)
    # assert V.shape[1] == W.shape[1]
    VW = np.vstack((V, W))
    return rank(VW) == rank(W)


def is_in_span(V, W):
    if V.ndim == 1:
        V = V.reshape(1, -1)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    assert V.shape[1] == W.shape[1]
    return is_in_span_nb(V, W)


def is_same_subspace(V, W):
    """Return True if V and W are bases for the same subspace"""
    if V.ndim == 1:
        V = V.reshape(1, -1)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    assert V.shape[1] == W.shape[1]
    VW = np.vstack((V, W))
    return rank(VW) == rank(W) and rank(V) == rank(W)


def matrix_period(A, maxiter=10000):
    assert A.shape[0] == A.shape[1]
    eye = np.eye(A.shape[0], dtype=A.dtype)
    if np.array_equal(A, eye):
        return 1
    AA = A.copy()
    for i in range(2, maxiter+2):
        AA = (A@AA) % 2
        if np.array_equal(AA, eye):
            return i 
    raise RuntimeError('maxiter exceeded')


def intersect(A, B):
    """Return a basis for the intersection of the subspaces
    spanned by the columns of A and B."""
    AB = np.hstack((A, B))
    kerAB = kernel(AB)
    return reduce_and_trim(kerAB[:,:A.shape[1]]@A.T % 2)

def difference(A, B):
    """Return a basis for the difference of the subspaces
    spanned by the columns of A and B, i.e., the subspace of 
    A that is linearly independent of B."""
    AB = np.hstack((A, B))
    imAB = image(AB)
    return reduce_and_trim(imAB[:,:A.shape[1]]@A.T % 2)