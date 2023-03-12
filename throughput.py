import numpy as np
import pickle
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from datetime import datetime

import lattice as lt 
import bipartite as bp


def throughput(tensor, n, N):
    params = {'phys_constr':True,
              'mode':'stab',
              'z_loc':0,
              'left_edge_constr':False,
              'right_edge_constr':False,
              'obc':False,
              'top_edge_constr':False,
              'bottom_edge_constr':False}
    lattice_symms = lt.lattice_symms(tensor, n, N, params)
    left_edge_symms = lattice_symms[:,lt.get_edge_mask(n, N, 'left')]
    right_edge_symms = lattice_symms[:,lt.get_edge_mask(n, N, 'right')]
    left_nonzero = np.sum(left_edge_symms, axis=1)
    right_nonzero = np.sum(right_edge_symms, axis=1)
    left_right_nonzero = np.logical_or(left_nonzero, right_nonzero)
    left_edge_symms = left_edge_symms[left_right_nonzero]
    right_edge_symms = right_edge_symms[left_right_nonzero]
    edge_symms = np.hstack((left_edge_symms, right_edge_symms))
    G, Gbar, Z = bp.canonical_form(edge_symms)
    return G.shape[0]


def throughput_grid(tensor, nmax, Nmax, progress=True):
    grid = np.zeros((nmax, Nmax), dtype=np.uint8)
    iter_i = range(nmax)
    for i in (tqdm(iter_i) if progress else iter_i):
        for j in range(Nmax):
            grid[i, j] = throughput(tensor, i+1, j+1)
    return grid


if __name__ == '__main__':
    with open('stabs5_reps.pkl', 'rb') as f:
        stabs5_reps = pickle.load(f)
        
    tp_grids = []
    for i in trange(len(stabs5_reps)):
        tp_grids.append(
            throughput_grid(stabs5_reps[i], 10, 10, progress=False))
        
    grids = np.unique(tp_grids, axis=0)
    grids = grids[np.argsort(grids.sum((1,2)))]

    n = 10
    cmap = plt.get_cmap('viridis', n+1)
    fig, ax = plt.subplots(2, 6, figsize=(12, 4))
    for i in range(12):
        _ = plt.sca(ax.flat[i])
        im = plt.imshow(grids[i+1], vmin=0-0.5, vmax=n+0.5, cmap=cmap)
        _ = plt.title('Class {}'.format(i+1))
        if i == 6:
            _ = plt.xticks(np.arange(1, n, 2), range(2, n+1, 2))
            _ = plt.yticks(np.arange(1, n, 2), range(2, n+1, 2))
            _ = plt.xlabel(r'$d$')
            _ = plt.ylabel(r'$n$')
        else:
            _ = plt.xticks([])
            _ = plt.yticks([])
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(0, n+1))
    cbar.set_label(r'$C(n,d)$')

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H:%M:%S")
    fname = 'tp_classes_{}x{}_'.format(n) + date_time + '.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')