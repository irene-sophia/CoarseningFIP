import networkx as nx
from sort_nodes import sort_nodes
from delete_duplicate_edges import delete_duplicate_edges
from check_connectivity_faster import check_connectivity_faster
from rulesets import rulesets
from delete_end_nodes import delete_end_nodes
from delete_self_loops import delete_self_loops
from add_link_neighbors import add_link_neighbors
from add_vertex_neighbors import add_vertex_neighbors


def combine_links(links, nodes, weights, params):
    # extract the necessary parameter values
    exempt_ids = params['exempt_ids']
    pruning = params['pruning']

    # combine links with one neighbor and similar speed
    nodes = sort_nodes(nodes)
    links, weights = delete_duplicate_edges(links, weights)
    G = nx.MultiDiGraph()
    G.add_edges_from([(link[0], link[1]) for link in links])

    checked_nodes = []
    for link in links:
        try:
            x = links[link]['removed_nodes']  # todo nodes of links?
            checked_nodes += x
        except KeyError:
            # links[link]['removed_nodes'] = []
            links[link]['removed_nodes'] = []

        try:
            x = links[link]['removed_links']  # todo nodes of links?
            checked_nodes += x
        except KeyError:
            links[link]['removed_links'] = [link]

    for node in nodes:
        try:
            x = nodes[node]['removed_nodes']
        except KeyError:
            nodes[node]['removed_nodes'] = [node]
    checked_nodes = list(set(checked_nodes))  # unique values

    unused_nodes = set(nodes) - set([link[0] for link in links] + [link[1] for link in links])
    for node in nodes.copy():
        if node not in list(G.nodes()):
            del nodes[node]
    G.remove_nodes_from(unused_nodes)
    # A = nx.adjacency_matrix(G)
    max_link_id = max([v['id'] for k, v in links.items() ])
    # initial_S = check_connectivity_faster(A)
    exempt_ids = sorted(exempt_ids)

    if pruning == 1:
        all_nodes = [node for node, _ in nodes.items() if nodes[node]['num_neighboring_nodes'] > 0]
    else:
        all_nodes = [node for node, _ in nodes.items() if nodes[node]['num_neighboring_nodes'] > 1]

    for node in all_nodes:
        if (node in exempt_ids) or (node in checked_nodes):
            pass
        else:
            pred = list(G.in_edges(node))
            succ = list(G.out_edges(node))
            all_link_ids = pred + succ
            checked_nodes.append(node)
            if not all_link_ids:
                pass
            else:
                links, nodes, weights, max_link_id, i = rulesets(links, nodes, weights, node, max_link_id, params, pred, succ)

    links, weights = delete_duplicate_edges(links, weights)
    G = nx.MultiDiGraph()
    G.add_edges_from([(link[0], link[1]) for link in links])
    unused_nodes = set(nodes) - set([link[0] for link in links] + [link[1] for link in links])
    G.remove_nodes_from(unused_nodes)
    for node in nodes.copy(): 
        if node not in list(G.nodes()):
            del nodes[node]
        # try:
        #     x = G[node]
        # except:
        #     del nodes[node]


    # A = nx.adjacency_matrix(G)
    # S_conn = check_connectivity_faster(A)  # bugs due to deprecation
    # if (S_conn == initial_S):
    #     pass
    # else:
    #     print('Links were pruned')

    links = add_link_neighbors(G, links)
    nodes = add_vertex_neighbors(G, nodes, links) 

    if pruning == 1:
        G, links, nodes, weights = delete_end_nodes(G, links, weights, nodes, exempt_ids)
        links, weights = delete_self_loops(links, weights, exempt_ids)

    links = add_link_neighbors(G, links)
    nodes = add_vertex_neighbors(G, nodes, links) 

    # also delete nodes that have no neighbors? (no that should be fixed by the check_connectivity func)

    return links, nodes, weights
