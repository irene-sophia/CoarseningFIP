from delete_duplicate_edges import delete_duplicate_edges


def delete_self_loops(links, weights, exempt_ids):

    # o_nodes = [link[0] for link in links]  # origin node = index 0
    # d_nodes = [link[1] for link in links] # destination node = index 1
    links = {k:v for k,v in links.items() if (k[0] != k[1] and k[0] not in exempt_ids)}

    # if not exempt_ids:
    #     links = {k:v for k,v in links.items() if k[0] != k[1]}
    # else:
    #     duplicates = [k[0] for k,v in links.items() if k[0] == k[1]]


    #     indexes = [i for i, (o, d) in enumerate(zip(o_nodes, d_nodes)) if o == d]
    #     indexo = [o in exempt_ids for o in o_nodes[indexes]]
    #     indexd = [d in exempt_ids for d in d_nodes[indexes]]
    #     c = [i or j for i, j in zip(indexo, indexd)]
    #     d = [i for i, j in zip(indexes, c) if not j]
    #     links = [links[i] for i in d]
    #     weights = [weights[i] for i in d]
    
    links, weights = delete_duplicate_edges(links, weights)
    
    return links, weights
