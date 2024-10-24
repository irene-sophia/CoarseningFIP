def add_vertex_neighbors(G, vertex, links):
    # This function adds the links neighbors of a vertex to the vertex table

    # all_o_nodes = [link[0] for link in links]
    # all_d_nodes = [link[1] for link in links]
    # all_o_nodes_sorted = sorted(all_o_nodes)
    # all_d_nodes_sorted = sorted(all_d_nodes)
    num_zeros = 0
    num_ones = 0
    for i in vertex:
        pred_nodes = list(G.predecessors(i))
        succ_nodes = list(G.successors(i))

        neighbors = pred_nodes + succ_nodes
        vertex[i]['num_neighboring_nodes'] = len(set(neighbors))

        # sucIDs = [index for index, node in enumerate(all_o_nodes_sorted) if vertex[i].id == node]
        # preIDs = [index for index, node in enumerate(all_d_nodes_sorted) if vertex[i].id == node]
        # vertex[i].predecessors = [links[preID].id for preID in preIDs]  # TODO: kijk of deze nodig zijn
        # vertex[i].successors = [links[sucID].id for sucID in sucIDs]  # TODO: kijk of deze nodig zijn

        # vertex[i].neighbor_length = len(vertex[i].predecessors) + len(vertex[i].successors)
    return vertex