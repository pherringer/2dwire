import numpy as np

import symplectic as sp

def canonical_form(S):
    assert not S.shape[1] % 2
    n = S.shape[1]//2
    S = [tuple(x) for x in S]
    G = []
    Gbar = []
    Z = []
    while len(S):
        v = S[0] 
        for w in S:
            if sp.bsp_single_numba(v[:n], w[:n]):
                G.append(v)
                Gbar.append(w)
                S.remove(v)
                S.remove(w)
                for i in range(len(S)):
                    if sp.bsp_single_numba(v[:n], S[i][:n]):
                        S[i] = sp.xor(S[i], w)
                    if sp.bsp_single_numba(w[:n], S[i][:n]):
                        S[i] = sp.xor(S[i], v)
                break
        else:
            Z.append(v)
            S.remove(v)
    G = np.array(G, dtype=np.uint8)
    Gbar = np.array(Gbar, dtype=np.uint8)
    Z = np.array(Z, dtype=np.uint8)
    return G, Gbar, Z

