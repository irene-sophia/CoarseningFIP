from ismember1D import ismember1D

# The `add_link_neighbors` function in Python is designed to add neighbor information to each link in a graph.
# It takes in one parameter: `links`, which is a list of link objects.
# Each link object is assumed to have at least two attributes: `o_node` (origin node) and `d_node` (destination node).

# The function starts by extracting all origin nodes and destination nodes from the `links` list
# and storing them in `all_o_nodes` and `all_d_nodes` respectively.
# It then sorts these lists and stores the sorted lists and their original indices in `all_o_nodes_sorted`,
# `sortOInds`, `all_d_nodes_sorted`, and `sortDInds`.

# The function then enters a loop that iterates over each link in `links`.
# For each link, it initializes an empty list `link.neighbors` to store the IDs of the link's neighbors.
# It then finds the indices of the link's origin and destination nodes in `all_o_nodes_sorted` and `all_d_nodes_sorted`
# using the `ismember1D` function. These indices are stored in `oo_inds`, `do_inds`, `od_inds`, and `dd_inds`.

# The function then creates a list `all_ids` that contains all these indices
# and removes duplicate indices from `all_ids` to create `unique_ids`.
# If `unique_ids` is not empty, the function enters another loop that iterates over each index in `unique_ids`.
# For each index, if it is not the same as the current link's index and it is not 0,
# the function adds the ID of the link at this index to `link.neighbors`.   # this id should be the actual node id not the enumerate bs there is now!

# Finally, the function returns the updated `links` list.
# In summary, this function adds neighbor information to each link in a graph
# by finding the links that have the same origin or destination node as the current link
# and adding their IDs to the current link's `neighbors` attribute.


def add_link_neighbors(G, links):
    # all_o_nodes = [link[0] for link in links]
    # all_d_nodes = [link[1] for link in links]
    # all_o_nodes_sorted, sortOInds = zip(*sorted(enumerate(all_o_nodes), key=lambda x: x[1]))  # TODO: not enumerate! get actual id
    # all_d_nodes_sorted, sortDInds = zip(*sorted(enumerate(all_d_nodes), key=lambda x: x[1]))
    
    for i, link in enumerate(links):
        # add neighbor info to each link
        links[link]['neighbors'] = []
        try:
            oo = [links[b]['id'] for b in list(G.out_edges(link[0])) if b[0] != b[1]]
        except: 
            oo = []
        try:
            od = [links[b]['id'] for b in list(G.in_edges(link[0])) if b[0] != b[1]]
        except: 
            od = []
        try:
            do = [links[b]['id'] for b in list(G.out_edges(link[1])) if b[0] != b[1]]
        except:
            do = []
        try:
            dd = [links[b]['id'] for b in list(G.in_edges(link[1])) if b[0] != b[1]]
        except:
            dd = []
            
        all_ids = oo + do + od + dd
        unique_ids = list(set(all_ids))

        if not unique_ids:
            continue
        else: 
            for j in unique_ids:
                if j != links[link]['id']:
                    links[link]['neighbors'].append(j)  # added id; see if this is good or you want the u,v pair

    
    return links