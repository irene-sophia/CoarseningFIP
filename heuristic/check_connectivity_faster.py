import networkx as nx

def check_connectivity_faster(G):
    # Gt = nx.eye(G.shape[0])
    # Gtt = G + Gt
    # # How many connected components?
    # _, _, r = nx.linalg.csgraph.connected_components(Gtt)
    # S = len(r) - 1
    # C = np.cumsum(np.bincount(r[:-1]))
    # C[p] = C[p]

    # S, _ = nx.algorithms.components.connected_components(G, connection='weak')
    S, _ = nx.algorithms.components.connected_components(G, connection='weak')

    return S