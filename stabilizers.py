import numpy as np
import itertools as it
from tqdm import tqdm
import pickle
from datetime import datetime
import os

import linalg
import symplectic as sp 

def num_stabs(n):
    """Number of n-qubit stabilizer states, up to sign.

    Args:
        n (int): Number of qubits.

    Returns:
        int: Number of n-qubit stabilizer states, up to sign.
    """
    return np.prod(2**np.arange(1, n+1)+1)


def to_tuple(arr):
    """Convert 2D array to tuple of tuples."""
    return tuple(map(tuple, arr))


def to_tuple_recursive(a):
    try:
        return tuple(to_tuple_recursive(i) for i in a)
    except TypeError:
        return a


def grow_isotropic_subspaces(subspaces, verbose=False, progress=False):
    """Takes a set of isotropic subspaces and returns the set of all
    isotropic subspaces with one more basis vector.

    Args:
        subspaces (set): Bases for isotropic subspaces, in RREF
        verbose (bool, optional): For debugging. Defaults to False.
        progress (bool, optional): Display progess bar. Defaults to False.

    Raises:
        RuntimeError: If the subspaces are already maximally isotropic.

    Returns:
        set: Bases for expanded isotropic subspaces.
    """
    V_test = np.array(next(iter(subspaces)))
    assert not V_test.shape[1] % 2
    if V_test.shape[0] == (V_test.shape[1]//2):
        raise RuntimeError('Subspaces are already maximum size')
    bigger_subspaces = set()
    iter_subspaces = iter(subspaces)
    iter_i = range(len(subspaces))
    for i in (tqdm(iter_i) if progress else iter_i):
        V_arr = np.array(next(iter_subspaces), dtype=np.uint8)
        if verbose and i == 0:
            print('V_arr', V_arr.shape)
        V_perp = linalg.kernel(sp.lambda_right(V_arr))
        if verbose and i == 0:
            print('V_perp', V_perp.shape)
        V_perp_indep = linalg.difference(V_perp.T, V_arr.T)
        if verbose and i == 0:
            print('V_perp_indep', V_perp_indep.shape)
        P = np.array(
            list(it.product((0, 1), repeat=V_perp_indep.shape[0])), 
            dtype=np.uint8)[1:]
        V_perp_indep_all = (P@V_perp_indep) % 2
        V_perp_indep_all = V_perp_indep_all.reshape(
            V_perp_indep_all.shape[0], 1, V_perp_indep_all.shape[1])
        if verbose and i == 0:
            print('V_perp_indep_all', V_perp_indep_all.shape)
        V_arr_repeat = np.repeat(
            V_arr.reshape(1, *V_arr.shape), V_perp_indep_all.shape[0], axis=0)
        if verbose and i == 0:
            print('V_arr_repeat', V_arr_repeat.shape)
        V_new_candidates = np.concatenate(
            (V_arr_repeat, V_perp_indep_all), axis=1)
        if verbose and i == 0:
            print('V_new_candidates', V_new_candidates.shape)
        for j, V_new in enumerate(V_new_candidates):
            if verbose and i == 0 and j == 0:
                print('V_new', V_new.shape)
            linalg.rref(V_new, in_place=True)
            V_new_tuple = to_tuple(V_new)
            if V_new_tuple not in bigger_subspaces:
                bigger_subspaces.add(V_new_tuple)
    return bigger_subspaces


def get_stabilizer_states(n, progress=False):
    """Generates all the n-qubit stabilizer states, up to sign. 
    Stabilizer states are represented by Lagrangian (maximally isotropic)
    subspaces of the binary symplectic vector space with dimension 2n. 
    The check vector indexing is (x_1, z_1, ..., x_n, z_n).

    Args:
        n (int): Number of qubits
        progress (bool, optional): Display progress bar. Defaults to False.

    Returns:
        array (3d): Binary matrix representation of all n-qubit stabilizer
            states, up to sign
    """
    subspaces = np.array(
        list(it.product((0, 1), repeat=2*n))).reshape(-1, 1, 2*n)[1:]
    iter_i = range(n-1)
    for i in (tqdm(iter_i) if progress else iter_i):
        subspaces = grow_isotropic_subspaces(subspaces, progress=progress)
    return np.array(list(subspaces), dtype=np.uint8)


def get_eq_classes(alist, gen_func, progress=True):
    """Returns list of class representative, list of class labels, list of full classes. Return type of gen_func must support == comparison"""
    full_classes = []
    class_examples = []
    class_labels = []
    iter_i = range(len(alist))
    for i in (tqdm(iter_i) if progress else iter_i):
        aclass = gen_func(alist[i])
        try:
            idx = full_classes.index(aclass)
            class_labels.append(idx)
        except ValueError:
            class_labels.append(len(full_classes))
            full_classes.append(aclass)
            class_examples.append(alist[i])
    return class_examples, class_labels, full_classes


def clifford_1():
    e = np.eye(2, dtype=int)  # I 
    h = np.array([[0, 1], [1, 0]])  # X <--> Z (Hadamard)
    s = np.array([[1, 0], [1, 1]])  # X <--> Y (Phase) 
    hs = h@s % 2  # X --> Y --> Z --> X
    sh = s@h % 2  # X --> Z --> Y --> X
    hsh = h@s@h % 2  # Y <--> Z
    return e, h, s, hs, sh, hsh


def clifford_perms_tensor_4(tensor):
    single_legs = np.split(tensor, tensor.shape[1]//2, axis=1)
    permuted_tensors = []
    cliff1 = clifford_1()
    for g, h in it.product(cliff1, repeat=2):
        temp = np.hstack((
                (g@single_legs[0].T).T % 2,
                (g@single_legs[1].T).T % 2,
                (h@single_legs[2].T).T % 2,
                (h@single_legs[3].T).T % 2,
            ))
        permuted_tensors.append(temp)
    return permuted_tensors


def clifford_perms_tensor_5(tensor, physical=False):
    single_legs = np.split(tensor, tensor.shape[1]//2, axis=1)
    permuted_tensors = []
    cliff1 = clifford_1()
    if physical:
        for g, h, k in it.product(cliff1, repeat=3):
            temp = np.hstack((
                    (g@single_legs[0].T).T % 2,
                    (g@single_legs[1].T).T % 2,
                    (h@single_legs[2].T).T % 2,
                    (h@single_legs[3].T).T % 2,
                    (k@single_legs[4].T).T % 2,
                ))
            permuted_tensors.append(temp)
    else:
        for g, h in it.product(cliff1, repeat=2):
            temp = np.hstack((
                (g@single_legs[0].T).T % 2,
                (g@single_legs[1].T).T % 2,
                (h@single_legs[2].T).T % 2,
                (h@single_legs[3].T).T % 2,
            ))
            permuted_tensors.append(np.hstack((temp, single_legs[4])))
    return permuted_tensors


def tensor_eq_cls_4(tensor):
    eq_cls = [linalg.rref(x) for x in clifford_perms_tensor_4(tensor)]
    eq_cls = np.unique(eq_cls, axis=0)
    return to_tuple_recursive(eq_cls)


def tensor_eq_cls_5(tensor):
    eq_cls = [linalg.rref(x) for x in clifford_perms_tensor_5(tensor)]
    eq_cls = np.unique(eq_cls, axis=0)
    return to_tuple_recursive(eq_cls)


if __name__ == '__main__':
    if os.path.exists('stabs5.pkl'):
        with open('stabs5.pkl', 'rb') as f:
            stabs = pickle.load(f)
        print('Loaded 5-qubit stabilizer states')
    else:
        print('Generating 5-qubit stabilizer states')
        stabs = get_stabilizer_states(5, progress=True)

        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y_%H-%M-%S")
        fname = 'stabs5_' + date_time + '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(stabs, f)

    print('Finding equivalence classes')
    reps, labels, classes = get_eq_classes(
        stabs, tensor_eq_cls_5)

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y_%H-%M-%S")
    fname_reps = 'stabs5_reps_' + date_time + '.pkl'
    fname_labels = 'stabs5_labels_' + date_time + '.pkl'
    fname_classes = 'stabs5_classes_' + date_time + '.pkl'
    with open(fname_reps, 'wb') as f:
        pickle.dump(reps, f)
    with open(fname_labels, 'wb') as f:
        pickle.dump(labels, f)
    with open(fname_classes, 'wb') as f:
        pickle.dump(classes, f)


