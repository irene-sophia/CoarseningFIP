import numpy as np
from shapely.wkt import loads

def collapse_node(links, nodes, weights, node_id, max_link_id, incoming_index, outgoing_index, pruning):
    # incoming/ outcoming are LINKS (indices) that are connected to the node we're collapsing 
    node_ids = []
    p, q = np.meshgrid(incoming_index, outgoing_index)  # combinations again??
    index_pairs = np.column_stack((p.flatten(), q.flatten()))
    # A[node_id, :] = 0  # let's see if we can do without A
    # A[:, node_id] = 0
    for i in range(index_pairs.shape[0]):
        # try:
        #     if (index_pairs == np.array([[5690, 5690], [5690, 5691]])).all():
        #         print('here')
        # except:
        #     pass

        if index_pairs[i,0] == index_pairs[i, 1]:
            continue
        try:
            o_node = [k[0] for k, v in links.items() if v['id'] == index_pairs[i, 0]][0]
            node_to_be_collapsed = [k[0] for k, v in links.items() if v['id'] == index_pairs[i, 1]][0]
            d_node = [k[1] for k, v in links.items() if v['id'] == index_pairs[i, 1]][0]
        except:
            continue  # TODO what happens here actually: trying to delete a non existing node?

        if (pruning == 0) or (o_node != d_node):
            # create new link info: 

            ### id, o_node, d_node, distance, weight, removed_nodes, removed_links,
            # rem_nodes = links[(o_node, node_to_be_collapsed)]['removed_nodes'] + links[(node_to_be_collapsed, d_node)]['removed_nodes'] + [node_id]
            #TODO: kijk of je die ids dan nog kan herleiden naar de originele links later
            #rem_links = links[(o_node, node_to_be_collapsed)]['removed_links'] + links[(node_to_be_collapsed, d_node)]['removed_links'] + [links[(o_node, node_to_be_collapsed)]['id'], links[(node_to_be_collapsed, d_node)]['id']]
            rem_links = (links[(o_node, node_to_be_collapsed)]['removed_links'] + links[(node_to_be_collapsed, d_node)]['removed_links'] +
                         [(o_node, node_to_be_collapsed),(node_to_be_collapsed, d_node)])

            if 'geometry' in links[(o_node, node_to_be_collapsed)]:
                if 'geometry' in links[(node_to_be_collapsed, d_node)]:
                    coords1 = str(links[(o_node, node_to_be_collapsed)]['geometry']).split('(')[1].split(')')[0]
                    coords2 = str(links[(node_to_be_collapsed, d_node)]['geometry']).split('(')[1].split(')')[0]
                    geomm = 'LINESTRING (' + coords1 + ', ' + coords2 + ')'
                    geom = loads(geomm)
                else:
                    geom = links[(o_node, node_to_be_collapsed)]['geometry']
            else:
                if 'geometry' in links[(node_to_be_collapsed, d_node)]:
                    geom = links[(node_to_be_collapsed, d_node)]['geometry']
                else:
                    continue
            
            # put removed node on closest node (o/d):
            closest_neighbor = o_node if links[(o_node, node_to_be_collapsed)]['length'] < links[(node_to_be_collapsed, d_node)]['length'] else d_node
            nodes[closest_neighbor]['removed_nodes'] = nodes[closest_neighbor]['removed_nodes'] + nodes[node_to_be_collapsed]['removed_nodes']

            links[(o_node, d_node)] = {'length': links[(o_node, node_to_be_collapsed)]['length'] + links[(node_to_be_collapsed, d_node)]['length'],
                                       'id': max_link_id+1, 
                                       'weight': np.mean([links[(o_node, node_to_be_collapsed)]['weight'], links[(node_to_be_collapsed, d_node)]['weight']]), 
                                    #    'removed_nodes': rem_nodes, 
                                       'removed_links': rem_links, 
                                       'travel_time': links[(o_node, node_to_be_collapsed)]['travel_time'] + links[(node_to_be_collapsed, d_node)]['travel_time'],
                                       'geometry': geom,
                                       }
            weights[max_link_id+1] = np.mean([links[(o_node, node_to_be_collapsed)]['weight'], links[(node_to_be_collapsed, d_node)]['weight']])
            
            max_link_id += 1
            
            # links.append(links[index_pairs[i, 0]])
            # links[-1].id = max_link_id
            # links[-1].d_node = links[index_pairs[i, 1]].d_node
            # links[-1].distance = links[index_pairs[i, 0]].distance + links[index_pairs[i, 1]].distance
            # links[-1].weight = np.mean([weights[index_pairs[i, 0]], weights[index_pairs[i, 1]]])
            # links[-1].removed_nodes = links[index_pairs[i, 0]].removed_nodes + links[index_pairs[i, 1]].removed_nodes + [node_id]
            # links[-1].removed_links = links[index_pairs[i, 0]].removed_links + links[index_pairs[i, 1]].removed_links + [links[index_pairs[i, 0]].id, links[index_pairs[i, 1]].id]
            # weights.append(links[-1].weight)
            # max_link_id += 1
            # A[links[-1].o_node, links[-1].d_node] = 1
            node_ids.extend([o_node, d_node])
    indexes = np.concatenate((p.flatten(), q.flatten()))


    links = {k: v for k, v in links.items() if v['id'] not in indexes}
    weights = {k: v for k, v in weights.items() if k not in indexes}
    node_ids.append(node_id)  # TODO: what are you for???
    return links, nodes, weights, max_link_id
