def sort_nodes(nodes):
    # sort the nodes for hierarchical contraction
    # new_nodes = sorted(nodes, key=lambda x: x[1]['num_neighboring_nodes'])  # when links = G.edges(data=True)
    new_nodes = dict(sorted(nodes.items(), key = lambda x: (x[1]["num_neighboring_nodes"])))
    # new_nodes = [nodes[i] for i in ind]
    return new_nodes
