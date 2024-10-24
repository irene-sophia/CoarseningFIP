from ismember1D import ismember1D
from lldistkm import lldistkm


def create_links_table(shape, vertex):
    # This function creates link table from the updated shape struct which
    # includes the vertex ids. add link id to the shape struct. add link
    # distance and neighbors to links.

    # create link table
    len_way = len(shape)
    all_links = []
    for i in range(len_way):
        o_nodes = shape[i]['vertex_ids'][0]
        d_nodes = shape[i]['vertex_ids'][-1]
        temp = [o_nodes, d_nodes]
        all_links.append(temp)

    all_links = [item for sublist in all_links for item in sublist]
    unique_links = list(set(tuple(link) for link in all_links))

    links = []
    for i, link in enumerate(unique_links):
        link_dict = {
            'id': i,
            'o_node': link[0],
            'd_node': link[1]
        }
        links.append(link_dict)

    # add link id to shape struct
    all_links = [[link['o_node'], link['d_node']] for link in links]
    all_link_ids = [link['id'] for link in links]
    all_links_sorted = sorted(all_links)

    # The last elements are not searched
    all_links_sorted.extend([(max(all_links[i]) * 10) for i in range(5)])

    for i in range(len(shape)):
        o_nodes = shape[i]['vertex_ids'][0]
        d_nodes = shape[i]['vertex_ids'][-1]
        temp = [o_nodes, d_nodes]
        index = all_links.index(temp)
        shape[i]['link_ids'] = all_link_ids[index]

    # add length and neighbors to links
    all_vertex = [v['id'] for v in vertex]
    all_vertex.extend([(max(v['id']) * 10) for v in range(5)])
    all_o_nodes = [link['o_node'] for link in links]
    all_d_nodes = [link['d_node'] for link in links]
    all_o_nodes_sorted = sorted(all_o_nodes)
    all_d_nodes_sorted = sorted(all_d_nodes)

    for link in links:
        o_index = all_vertex.index(link['o_node'])
        d_index = all_vertex.index(link['d_node'])
        latlon1 = vertex[o_index]['coordinates']
        latlon2 = vertex[d_index]['coordinates']
        d1km = lldistkm(latlon1, latlon2)
        link['distance'] = d1km

        link['neighbors'] = []
        oo_inds = ismember1D(link['o_node'], all_o_nodes_sorted)
        do_inds = ismember1D(link['o_node'], all_d_nodes_sorted)
        od_inds = ismember1D(link['d_node'], all_o_nodes_sorted)
        dd_inds = ismember1D(link['d_node'], all_d_nodes_sorted)
        all_ids = [oo_inds, do_inds, od_inds, dd_inds]
        unique_ids = list(set(all_ids))
        if not unique_ids:
            continue
        else:
            for j in range(len(unique_ids)):
                if unique_ids[j] != link['id'] and unique_ids[j] != 0:
                    link['neighbors'].append(links[unique_ids[j]]['id'])

    return shape, links