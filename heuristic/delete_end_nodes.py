def delete_end_nodes(G, links, weights, nodes, exempt_ids):
    # remove dead ends (no successor nodes) & put info on closest pred node
    all_nodes = {k:v for k, v in nodes.items() if v['num_neighboring_nodes'] == 1}
    if exempt_ids:
        all_nodes = {k:v for k, v in all_nodes.items() if k not in exempt_ids}

    # remove links
    ## find connected links
    for node in all_nodes.keys():
        #put rem nodes & rem links in neighbor
        pred_nodes = list(G.predecessors(node))
        succ_nodes = list(G.successors(node))

        if node in exempt_ids:  # if exempt, skip
            continue

        elif len(pred_nodes) + len(succ_nodes) == 0:  # floating node with no neighbors
            #remove node
            nodes = {k:v for k, v in nodes.items() if k != node}
            G.remove_node(node)

        elif len(set(succ_nodes)) == 0:  # no successor nodes, but could have mutliple pred nodes like a -> b <- c (rem b)
            # put info on closest pred node
            nearest_pred = min(pred_nodes, key=lambda x: links[(x, node)]['length'])
            nodes[nearest_pred]['removed_nodes'] = nodes[nearest_pred]['removed_nodes'] + nodes[node]['removed_nodes']

            # remove links
            pred_link = list(G.in_edges(node))
            succ_link = list(G.out_edges(node))
            links = {k:v for k, v in links.items() if k not in pred_link + succ_link}

            #remove node
            nodes = {k:v for k, v in nodes.items() if k != node}
            G.remove_node(node)

        else:  # dead end/culdesac, but could turn around
            assert len(set(pred_nodes + succ_nodes)) == 1

            neighbor = (pred_nodes + succ_nodes)[0]
            nodes[neighbor]['removed_nodes'] = nodes[neighbor]['removed_nodes'] + nodes[node]['removed_nodes']

            #remove links
            pred = list(G.in_edges(node))
            succ = list(G.out_edges(node))
            links = {k:v for k, v in links.items() if k not in pred + succ}

            #remove node
            nodes = {k:v for k, v in nodes.items() if k != node}
            G.remove_node(node)

    # weights = [weight for i, weight in enumerate(weights) if i not in index]
    return G, links, nodes, weights
