import numpy as np
from scipy.sparse import bsr_matrix, lil_matrix, block_diag

import linalg

"""Indexing: lattice position (row major), tensor leg (left, right, top, 
bottom, physical), (X, Z). For example, the (0, 0) entry of an array is the X component of the left leg of the tensor at position (0, 0)."""

# Indices that map tensor legs to check vector
LEFT_X = 0
LEFT_Z = 1
RIGHT_X = 2
RIGHT_Z = 3
TOP_X = 4
TOP_Z = 5
BOTTOM_X = 6
BOTTOM_Z = 7
PHYS_X = 8
PHYS_Z = 9

# Length of check vector for a single tensor
LEN = 10


def get_horiz_leg_constr(m, n):
    width = m*n*LEN 
    if n > 1:
        first_two_horiz_x = np.zeros(width, dtype=np.uint8)
        first_two_horiz_x[[RIGHT_X, LEN+LEFT_X]] = 1
        first_row_horiz_x = np.vstack(
            [np.roll(first_two_horiz_x, i*LEN) for i in range(n-1)])
        full_grid_horiz_x = np.vstack(
            [np.roll(first_row_horiz_x, i*n*LEN, axis=1) for i in range(m)])
        full_grid_horiz_xz = np.vstack(
            [full_grid_horiz_x, np.roll(full_grid_horiz_x, 1, axis=1)])
    else:
        full_grid_horiz_xz = []
    return full_grid_horiz_xz


def get_vert_leg_constr(m, n, obc=False):
    width = m*n*LEN
    if m > 1:
        first_two_vert_x = np.zeros(width, dtype=np.uint8)
        first_two_vert_x[[BOTTOM_X, n*LEN+TOP_X]] = 1
        first_col_vert_x = np.vstack(
            [np.roll(first_two_vert_x, i*n*LEN) for i in range(m-int(obc))])
    else:
        first_col_vert_x = np.zeros(width, dtype=np.uint8)
        first_col_vert_x[[BOTTOM_X, TOP_X]] = 1
        first_col_vert_x = first_col_vert_x.reshape(1, -1)
    full_grid_vert_x = np.vstack(
        [np.roll(first_col_vert_x, i*LEN, axis=1) for i in range(n)])
    full_grid_vert_xz = np.vstack(
        [full_grid_vert_x, np.roll(full_grid_vert_x, 1, axis=1)])
    return full_grid_vert_xz


def get_phys_leg_constr(m, n, z_loc=0, mode='cone'):
    width = m*n*LEN
    if mode == 'loop':
        # Action on physical legs must be trivial
        full_grid_phys_xz = np.zeros((2*m*n, width), dtype=np.uint8)
        for i in range(m*n):
            full_grid_phys_xz[2*i, i*LEN+PHYS_X] = 1
            full_grid_phys_xz[2*i+1, i*LEN+PHYS_Z] = 1
    elif mode == 'cone':  
        # Action on physical legs must commute with Z at z_loc, and
        # commute with X everywhere else
        assert z_loc >= 0    
        full_grid_phys_xz = np.zeros((m*n, width), dtype=np.uint8)
        for i in range(m*n):
            if i == z_loc:
                full_grid_phys_xz[i, i*LEN+PHYS_X] = 1
            else:
                full_grid_phys_xz[i, i*LEN+PHYS_Z] = 1
    elif mode == 'stab':  
        # Action on physical legs must commute with X
        full_grid_phys_xz = np.zeros((m*n, width), dtype=np.uint8)
        for i in range(m*n):
            full_grid_phys_xz[i, i*LEN+PHYS_Z] = 1
    elif mode == 'z-type':
        # Action on physical legs must commute with Z
        full_grid_phys_xz = np.zeros((m*n, width), dtype=np.uint8)
        for i in range(m*n):
            full_grid_phys_xz[i, i*LEN+PHYS_X] = 1
    elif mode == 'id_cone':
        # Action on physical legs must commute with Z at z_loc, and
        # be trivial everywhere else
        assert z_loc >= 0    
        full_grid_phys_xz = np.zeros((2*m*n, width), dtype=np.uint8)
        for i in range(m*n):
            if i == z_loc:
                full_grid_phys_xz[2*i, i*LEN+PHYS_X] = 1
            else:
                full_grid_phys_xz[2*i, i*LEN+PHYS_X] = 1
                full_grid_phys_xz[2*i+1, i*LEN+PHYS_X] = 1
    else:
        raise ValueError('invalid mode')
    return full_grid_phys_xz


def get_edge_constr(m, n, side='left'):
    if side == 'left':
        return get_horiz_edge_constr(m, n, 'left')
    if side == 'right':
        return get_horiz_edge_constr(m, n, 'right')
    if side == 'top':
        return get_vert_edge_constr(m, n, 'top')
    if side == 'bottom':
        return get_vert_edge_constr(m, n, 'bottom')
    else:
        raise ValueError(
            'Side must be one of left, right, top, or bottom')


def get_horiz_edge_constr(m, n, side='left'):
    width = m*n*LEN
    init_x = np.zeros(width, dtype=np.uint8)
    if side == 'left':
        init_x[LEFT_X] = 1
    elif side == 'right':
        init_x[-LEN+RIGHT_X] = 1
    else:
        raise ValueError('side must be left or right')
    column_x = np.vstack(
        [np.roll(init_x, i*n*LEN) for i in range(m)])
    column_xz = np.vstack(
        [column_x, np.roll(column_x, 1, axis=1)])
    return column_xz


def get_vert_edge_constr(m, n, side='top'):
    width = m*n*LEN
    init_x = np.zeros(width, dtype=np.uint8)
    if side == 'top':
        init_x[TOP_X] = 1
    elif side == 'bottom':
        init_x[-n*LEN+BOTTOM_X] = 1
    else:
        raise ValueError('side must be left or right')
    row_x = np.vstack(
        [np.roll(init_x, i*LEN) for i in range(n)])
    row_xz = np.vstack(
        [row_x, np.roll(row_x, 1, axis=1)])
    return row_xz


def get_constr_mat_cvec(m, n, params=None):
    if params == None:
        params = {'phys_constr':False,
                  'mode':'cone',
                  'z_loc':0,
                  'left_edge_constr':False,
                  'right_edge_constr':False,
                  'obc':False,
                  'top_edge_constr':False,
                  'bottom_edge_constr':False
                  }
    constr = []
    if params['phys_constr']:
        phys = get_phys_leg_constr(
        m, n, z_loc=params['z_loc'], mode=params['mode'])
        constr.append(phys)
    if params['left_edge_constr']:
        left_edge = get_edge_constr(m, n, side='left')
        constr.append(left_edge)
    if params['right_edge_constr']:
        right_edge = get_edge_constr(m, n, side='right')
        constr.append(right_edge)
    # Horizontal and vertical leg constraints
    horiz = get_horiz_leg_constr(m, n)
    if len(horiz):
        constr.append(horiz)
    vert = get_vert_leg_constr(m, n, obc=params['obc'])
    if len(vert):
        constr.append(vert)
    if params['top_edge_constr']:
        assert params['obc']
        top_edge = get_edge_constr(m, n, side='top')
        constr.append(top_edge)
    if params['bottom_edge_constr']:
        assert params['obc']
        bottom_edge = get_edge_constr(m, n, side='bottom')
        constr.append(bottom_edge)
    # Stack all constraints
    return np.vstack(constr)


def get_gc_mat(cm_single, m, n):
    assert cm_single.ndim == 2
    assert cm_single.dtype == np.uint8
    g_to_c = np.block(
        np.kron(np.eye(m*n, dtype=np.uint8), cm_single.T))
    return bsr_matrix(g_to_c, blocksize=(cm_single.shape[1], cm_single.shape[0]))


def get_gc_mat_sparse(cm_single, m, n):
    assert cm_single.ndim == 2
    assert cm_single.dtype == np.uint8
    mats = np.broadcast_to(cm_single.T, (m*n, cm_single.shape[1], cm_single.shape[0]))
    return block_diag(mats, format='bsr')


def lattice_symms(
    cm_single, m, n, params=None):
    assert cm_single.ndim == 2
    assert cm_single.dtype == np.uint8

    constr_c = get_constr_mat_cvec(m, n, params=params)
    constr_c = lil_matrix(constr_c)
    constr_c = constr_c.tocsr()

    g_to_c = get_gc_mat_sparse(cm_single, m, n)

    constr_g = constr_c.dot(g_to_c).toarray() % 2
    ker_g = linalg.kernel(constr_g)
    ker_g = lil_matrix(ker_g)
    ker_g = ker_g.tocsr()

    ker_c = ker_g.dot(g_to_c.T).toarray() % 2
    return ker_c


def get_column_mask(m, n, k):
    """Mask for the k-th column of horizontal legs. The left edge
    is column 0, the right edge is column n."""
    mask = np.zeros((m*n, LEN), dtype=bool)
    if k == 0:
        for i in range(k, m*n, n):
            mask[i, [LEFT_X, LEFT_Z]] = 1
    elif k > 0 and k <= n:
        for i in range(k-1, m*n, n):
            mask[i, [RIGHT_X, RIGHT_Z]] = 1
    else:
        raise ValueError('must have 0 <= k <= n')
    return mask.flatten()


def get_edge_mask(m, n, side='left'):
    if side == 'left':
        return get_column_mask(m, n, 0)
    elif side == 'right':
        return get_column_mask(m, n, n)
    elif side == 'both':
        return get_column_mask(m, n, 0) + get_column_mask(m, n, n)


def get_phys_mask(m, n):
    mask = np.zeros((m*n, LEN), dtype=bool)
    mask[:, [PHYS_X, PHYS_Z]] = 1
    return mask.flatten()
    