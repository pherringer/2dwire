import numpy as np
import itertools as it

import linalg


idx_to_row = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.uint8)
row_to_idx = np.array([[0, 3], [1, 2]], dtype=np.uint8)

str_to_vec = {'I':(0, 0), 'X':(1, 0), 'Y':(1, 1), 'Z':(0, 1)}
vec_to_str = {v:k for k, v in str_to_vec.items()}


def str_to_cvec(pauli_str):
    """Converts a string to a check vector"""
    result = np.zeros(2*len(pauli_str), dtype=np.uint8)
    for i in range(len(pauli_str)):
        result[2*i:2*i+2] = str_to_vec[pauli_str[i]]
    return result


def str_arr_to_cmat(str_arr):
    """Converts a list or 1D array of strings to a check matrix"""
    result = np.zeros(
        (len(str_arr), 2*len(str_arr[0])), dtype=np.uint8)
    for i in range(len(str_arr)):
        result[i] = str_to_cvec(str_arr[i])
    return result


def cvec_to_str(v):
    assert not len(v) % 2
    result = ''
    for i in range(len(v)//2):
        result += vec_to_str[tuple(v[2*i:2*i+2])]
    return result


def cmat_to_str_arr(V):
    assert not V.shape[1] % 2
    result = []
    for i in range(len(V)):
        result.append(cvec_to_str(V[i]))
    return result 


def cvec_to_idx(v):
    assert not len(v) % 2
    result = np.zeros(len(v)//2, dtype=np.uint8)
    for i in range(len(v)//2):
        result[i] = row_to_idx[v[2*i], v[2*i+1]]
    return result


def cmat_to_idx_arr(V):
    assert not V.shape[1] % 2
    result = np.zeros((V.shape[0], V.shape[1]//2), dtype=np.uint8)
    for i in range(V.shape[0]):
        result[i] = cvec_to_idx(V[i])
    return result


def gen_group(G):
    assert linalg.rank(G) == G.shape[0]
    # All possible combinations of generators
    M = np.array([x for x in it.product((0, 1), repeat=G.shape[0])], dtype=np.uint8)
    # (G.T@M.T).T = M@G
    return (M@G) % 2


def cluster_group():
    gen_str = ['XZXZ', 'IXZI', 'ZIZI', 'ZIIX']
    gen_cmat = str_arr_to_cmat(gen_str)
    return gen_str, gen_cmat


def ghz_group():
    gen_str = ['XXXX', 'IZZI', 'ZIZI', 'ZIIZ']
    gen_cmat = str_arr_to_cmat(gen_str)
    return gen_str, gen_cmat