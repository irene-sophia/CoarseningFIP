import numpy as np

# The provided MATLAB function delete_duplicate_edges is designed to remove duplicate edges from a graph.
# It takes two inputs: links and weights. links is an array of structures where each structure represents an edge in the graph.
# Each structure in links has the following fields: o_node (origin node), d_node (destination node), distance (distance between the nodes),
# removed_nodes (nodes removed from the edge), and removed_links (links removed from the edge). weights is an array that represents the weight of each edge.

# The function begins by extracting the origin and destination nodes from the links array and storing them in o_nodes and d_nodes respectively.
# It then creates a 2D array all_pairs where each column represents an edge (origin node and destination node).

# The unique function is used to find the unique columns in all_pairs (i.e., the unique edges in the graph).
# It returns the indices of these unique edges in indexes. The setdiff function is then used to find the indices of the duplicate edges,
# which are stored in duplicate_ind.

# The function then enters a loop that iterates over each duplicate edge.
# For each duplicate edge, it finds all the edges in links that have the same origin and destination nodes.
# It then finds the minimum distance and maximum weight among these edges and assigns these values to the new edge.
# It also assigns the removed nodes and links from these edges to the new edge.

# Finally, the function updates the links and weights arrays to only include the unique edges (i.e., it removes the duplicate edges).
# It does this by indexing links and weights with indexes, which contains the indices of the unique edges.

# In summary, this function removes duplicate edges from a graph by finding all edges with the same origin and destination nodes,
# replacing them with a single edge that has the minimum distance and maximum weight among the duplicates,
# and updating the links and weights arrays to only include the unique edges.


def delete_duplicate_edges(links, weights):
    # copilot code
    o_nodes = [link[0] for link in links]  # origin node = index 0
    d_nodes = [link[1] for link in links] # destination node = index 1
    all_pairs = [[link[0], link[1]] for link in links]
    _, indexes = np.unique(all_pairs, axis=0, return_index=True)
    #find duplicate edges (TODO: is this really the best way lol)
    duplicate_ind = np.setdiff1d(np.arange(len(links)), indexes)  # data in A that is not in B
    for i in duplicate_ind:
        ind = i
        o_node = list(links)[ind][0]
        d_node = list(links)[ind][0]
        index = ((o_nodes == o_node) & (d_nodes == d_node))
        min_distance = min([link['distance'] for link in links[index]])
        max_weight = max(weights[index])
        rem_nodes = [link['removed_nodes'] for link in links[index]]
        rem_links = [link['removed_links'] for link in links[index]]
        # overwrite the data in all links with the same origin and destination nodes
        for link in links[index]:
            link['distance'] = min_distance
            link['removed_nodes'] = rem_nodes
            link['removed_links'] = rem_links
        weights[index] = max_weight
    # remove duplicates
    links = [links[i] for i in indexes]
    weights = [weights[i] for i in indexes]
    return links, weights

def delete_duplicate_edges(links, weights):
    o_nodes = [link[0] for link in links]  # origin node = index 0
    d_nodes = [link[1] for link in links] # destination node = index 1
    all_pairs = [[link[0], link[1]] for link in links]
    duplicates = [x for x in all_pairs if all_pairs.count(x) > 1]  #find duplicates
    duplicates = list(set(map(tuple, duplicates)))  # remove double entries
    for u,v in duplicates:
        # get edge with the lowest distance
        # assign highest weight only to all edges (weight = road type)
        # save removed nodes and links in the preserved edge
        # TODO: also preserve fug prob here?
        # del links[u,v]
        # add new edge with lowest distance and highest weight
        pass
    return links, weights