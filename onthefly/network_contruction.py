import networkx as nx


def construct_network(G_orig, routes_labeled, police_start):

    # get fugitive nodes set
    fugitive_nodes = set()
    for r, sublist in enumerate(routes_labeled):
        fugitive_nodes.update(set(sublist.values()))
    # get units node sets
    node_set = set()
    for pol_node in police_start:
        for fug_node in fugitive_nodes:
            try:
                node_set.update(nx.shortest_path(G_orig, pol_node, fug_node))
            except:
                print(pol_node, fug_node)

    # reconstruct graph
    G = G_orig.subgraph(nodes=node_set)
    return G


if __name__ == '__main__':
    pass