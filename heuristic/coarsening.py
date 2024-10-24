from combine_links import combine_links

def coarsening(links, vertex, weights, params):
    iterations = params['iterations']
    new_links = links
    new_vertex = vertex
    new_weights = weights
    flag = 1
    prev_links = new_links
    iter = 0
    exempt_ids_in_G = [idx for idx in params['exempt_ids'] if idx in new_vertex]
    while flag == 1 and iter < iterations:
        new_links, new_vertex, new_weights = combine_links(new_links, new_vertex, new_weights, params)
        # used_nodes = [node for node in new_vertex if node.neighbor_length > 0]
        # used_nodes = [node for node in new_vertex if new_vertex[node]['num_neighboring_nodes'] > 0]
        # if len(prev_links) == len(new_links) or len(used_nodes) <= 20:
        if len(prev_links) == len(new_links):
            flag = 0
            print('Coarsening finished after {} iterations'.format(iter))
            print(f'Original number of links: {len(links)}, New number of links: {len(new_links)}; link reduction of {round((len(links) - len(new_links))/len(links), 2)}.')
            print(f'Original number of nodes: {len(vertex)}, New number of nodes: {len(new_vertex)}; node reduction of {round((len(vertex) - len(new_vertex))/len(vertex), 2)}.')
        else:
            flag = 1
        prev_links = new_links
        iter += 1


    return new_links, new_vertex, new_weights

