# create the coarsened links and nodes from the link table and node table
# created from shape files. The network of Amsterdam is saved in folder as
# network-reduction.mat. You can create your own network in 
# main_create_network.py script file

# Coarsening Function
# new_links, new_vertex, new_weights = coarsening(links, vertex, weights, exempt_ids, pruning, threshold, iterations, constraint_links, flag_intersection)
# Parameters
# flag_study_area   - default is 0; 0 if you need to coarsen the whole   
#                     area (exempt_ids is empty), 1 for the study area 
#                     application where all nodes in the study area  
#                     (defined in exempt_ids) are preserved. 
# flag_intersection - default is 0; 0 if you want to coarsen all nodes, 
#                     1 for the intersection application where all nodes  
#                     other than the intersections are collapsed.
# constraint_links  - default is 1; 0 if you need to reduce the number of 
#                     nodes, 1 if you need to reduce the number of links
# pruning           - default is 0; 0 for pruning disabled, 1 for pruning enabled
# threshold         - 0 is the minimum threshold, maximum is variance(1,36)
#                     in case of using road type as the weights
# iterations        - 1 is the minimum threshold, set a high value for 
#                     maximum iterations. The iterations will be stopped 
#                     automatically when it converges.

import numpy as np
import networkx as nx
import osmnx as ox
import pickle

from shapely.wkt import loads

from coarsening import coarsening
from plot_network import plot_network
from define_study_area import define_study_area
from main_create_network import city_graph
from cut_graph_to_boundaries import cut_graph


# load network or create network
# main_create_network                                                       # <- uncomment to create Amsterdam network and then load network
# load('network.mat')
for weight_type in ['type']:
# for weight_type in ['type', 'betweenness']:
    for city in ['Utrecht', 'Winterswijk', 'Manhattan', 'Amsterdam', 'Rotterdam']:
    # for city in ['Winterswijk']:
    # for city in ['Winterswijk', 'Amsterdam', 'Manhattan']:
    # for city in ['Rotterdam']:
        # for pruning in [1]:
        #     for iterations in [1000]:
        #         for threshold in [1000]:
        for pruning in [1, 0]:
            for iterations in [1, 1000]:
                for threshold in [0, 1000]:
                    G, _, _ = city_graph(city, 0)
                    if city != 'Amsterdam':
                        G = cut_graph(G, city)

                    print(city, pruning, iterations, threshold, 'number of nodes original graph: ', len(G.nodes()))
                    num_neighbors = {i: len(G.adj[i]) for i in G.nodes()}
                    nx.set_node_attributes(G, num_neighbors, 'num_neighboring_nodes')
                    # links = G.edges(data=True)
                    # vertex = G.nodes(data=True)
                    links = {(u,v): data for u,v,data in G.edges(data=True)}
                    for id, link in enumerate(links):
                        links[link]['id'] = id
                    vertex = {u: data for u,data in G.nodes(data=True)}

                    if weight_type == 'type':
                        # initialize weights. Here the weights are the road type
                        types = []
                        for (u, v), data in links.items():
                            if type(data['highway']) == list:
                                types.append(data['highway'][0])
                            else:
                                types.append(data['highway'])

                        # weight dict
                        weight_dict = {'motorway': 1,
                                       'trunk': 2,
                                       'primary': 3,
                                       'secondary': 4,
                                       'tertiary': 5,
                                       'unclassified': 6,
                                       'residential': 7,
                                       'service': 8,
                                       'motorway_link': 9,
                                       'trunk_link': 10,
                                       'primary_link': 11,
                                       'secondary_link': 12,
                                       'tertiary_link': 13,
                                       'living_street': 14,
                                       'pedestrian': 15,
                                       'track': 16,
                                       'bus_guideway': 17,
                                       'raceway': 18,
                                       'road': 19,
                                       'busway': 20,
                                       'footway': 21,
                                       'bridleway': 22,
                                       'steps': 23,
                                       'path': 24,
                                       'cycleway': 25,
                                       'proposed': 26,
                                       'construction': 27,
                                       'bus_stop': 28,
                                       'crossing': 29,
                                       'elevator': 30,
                                       'emergency_access_point': 31,
                                       'escape': 32,
                                       'give_way': 33,
                                       'mini_roundabout': 34,
                                       'motorway_junction': 35,
                                       'passing_place': 36,
                                       'rest_area': 37
                                       }

                        #retrieve weights on road type
                        weights = {i: (1/weight_dict[x]) for i, x in enumerate(types)}
                        for id, link in enumerate(links):
                            links[link]['weight'] = weights[id]

                        threshold_ = threshold

                    if weight_type == 'betweenness':
                        # weights: betweenness centrality
                        with open(f'networks/betweenness_centrality_{city}_notnorm.pkl', 'rb') as f:
                            betweenness_centrality = pickle.load(f)
                        weights = {i: bt for i, bt in enumerate(list(betweenness_centrality.values()))}
                        for i, link in enumerate(links):
                            links[link]['weight'] = weights[i]

                        if threshold == 1000:
                            threshold_ = max(betweenness_centrality.values()) * max(betweenness_centrality.values())
                        else:
                            threshold_ = threshold

                    with open(f'networks/escape_nodes_{city}.pkl', 'rb') as f:
                        escape_nodes_ = pickle.load(f)
                    with open(f'networks/fugitive_start_{city}.pkl', 'rb') as f:
                        fugitive_start = pickle.load(f)
                    with open(f'networks/start_police_{city}.pkl', 'rb') as f:
                        police_start = pickle.load(f)

                    # make sure all escape nodes are in G and have a path from fug start
                    escape_nodes = []
                    for node in escape_nodes_:
                        if node in G.nodes():
                            if nx.has_path(G, source=fugitive_start, target=node):
                                escape_nodes.append(node)
                    # print(len(escape_nodes_), len(escape_nodes))

                    # initialize the parameters
                    # exempt ids could be exit nodes, fug start, pol start


                    params = {
                        'flag_study_area': 0,
                        'flag_intersection': 0,
                        'exempt_ids': escape_nodes + [fugitive_start] + police_start,
                        # 'exempt_ids': [],
                        'pruning': pruning,
                        'threshold': threshold_,
                        'iterations': iterations,
                        'constraint_links': 1
                    }

                    # if params['flag_study_area'] == 1:
                    #     # load('study_area_boundary.mat')                                        # sample study area boundary of amsterdam
                    #     params['exempt_ids'] = define_study_area(vertex, lat_min, lat_max, lon_min, lon_max)

                    # coarsening framework
                    new_links, new_vertex, new_weights = coarsening(links, vertex, weights, params)

                    exempt_ids_in_G = [idx for idx in params['exempt_ids'] if idx in vertex]
                    assert set(exempt_ids_in_G).issubset(G.nodes())
                    # if not set(exempt_ids_in_G).issubset(new_vertex):
                    #     print('not all exempt ids in coarsened_G')

                    # reconstruct G from links & nodes:
                    # G_from_links = nx.from_edgelist(new_links)  # when links = G.edges(data=True)
                    # datas = {}
                    # for v, data in G.nodes(data=True):  # G: old graph
                    #     datas[v] = data
                    # nx.set_node_attributes(G_from_links, datas)

                    # G_from_links = nx.DiGraph(new_links)

                    for link in new_links:
                        if 'geometry' in new_links[link].keys():
                            if type(new_links[link]['geometry']) == str:
                                new_links[link]['geometry'] = loads(new_links[link]['geometry'])

                    new_links_mdg = {}
                    for (u, v), data in new_links.items():
                        new_links_mdg[u, v, 0] = data

                    G_from_links = nx.from_edgelist(new_links, create_using=nx.MultiDiGraph)
                    nx.set_edge_attributes(G_from_links, new_links_mdg)
                    nx.set_node_attributes(G_from_links, new_vertex)

                    # Save results
                    # np.savez('network-reduction.npz', new_links=new_links, new_vertex=new_vertex, new_weights=new_weights)
                    # with open(f"results/coarsened_network_{city}_pruning{pruning}_iter{iterations}.pkl", 'wb') as f:
                    #     pickle.dump(G_from_links, f)
                    # print('number of nodes before adding detailed areas: ', len(G_from_links.nodes()))
                    # G, _, _ = city_graph(city, 0)

                    # add detailed graph around start locs
                    cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|secondary|tertiary|residential"]'
                    if city == 'Amsterdam':
                        cf = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|secondary"]'

                    x = nx.get_node_attributes(G, "x")
                    y = nx.get_node_attributes(G, "y")
                    # detailed_graphs = []
                    # if city == 'Winterswijk':
                    #     for node in escape_nodes:
                    #         G_node = ox.graph.graph_from_point((y[node], x[node]), dist=2000, custom_filter=cf)
                    #         detailed_graphs.append(G_node)
                    # for node in police_start + [fugitive_start]:
                    #     # if city == 'Winterswijk' and node in escape_nodes:
                    #     #     G_node = ox.graph.graph_from_point((y[node], x[node]), dist=2000, custom_filter=cf)
                    #     # if city == 'Amsterdam' and node == 2362166488:
                    #     #     G_node = ox.graph.graph_from_point((y[node], x[node]), dist=3000, custom_filter=cf)
                    #     # else:
                    #     try:
                    #         G_node = ox.graph.graph_from_point((y[node],x[node]), dist=500, custom_filter=cf)
                    #         # print('yes', node)
                    #     except:
                    #         # print(node, 'cannot be used to construct a graph')
                    #         pass
                    #
                    #     detailed_graphs.append(G_node)
                    # G_comb_coarsened = nx.compose_all([G_from_links]+detailed_graphs)

                    G_comb_coarsened = G_from_links.copy()
                    G_comb_coarsened.graph['crs'] = 4326
                    # cut to boundaries (function)
                    if city != 'Amsterdam':
                        G_comb_coarsened = cut_graph(G_comb_coarsened, city)

                    cc = max(nx.weakly_connected_components(G), key=len)
                    for node in G_comb_coarsened.copy().nodes():
                        if node not in cc:
                            G_comb_coarsened.remove_node(node)

                    en_without_path = []
                    for escape_node in escape_nodes + police_start + [fugitive_start]:
                        if escape_node not in G_comb_coarsened.nodes():
                            my_dict = {k:v for k, v in nx.shortest_path_length(G, target=escape_node, weight='travel_time').items() if k in G_comb_coarsened.nodes}
                            if my_dict:
                                closest_node = min(my_dict, key=my_dict.get)
                                sp = nx.shortest_path(G, closest_node, escape_node, weight='travel_time')
                                assert sp[-1] == escape_node
                                edges = [(sp[i], sp[i + 1]) for i in range(len(sp) - 1)]
                                for edge in edges:
                                    G_comb_coarsened.add_edge(edge[0], edge[1], key=0,
                                                              travel_time=G[edge[0]][edge[1]][0]['travel_time'],
                                                              geometry=G[edge[0]][edge[1]][0]['geometry'])
                                    G_comb_coarsened.add_edge(edge[1], edge[0], key=0,
                                                              travel_time=G[edge[0]][edge[1]][0]['travel_time'],
                                                              geometry=G[edge[0]][edge[1]][0]['geometry'])
                                for node in sp:
                                    if not G_comb_coarsened.nodes(data=True)[node]:
                                        nx.set_node_attributes(G_comb_coarsened, {node: {'x': x[node], 'y': y[node]}})

                                # print(len(edges))
                                # print(len(G_comb_coarsened.nodes))
                                if escape_node not in G_comb_coarsened.nodes():
                                    print('added ', escape_node, 'unsuccessfully')
                            else:
                                # print(escape_node, 'has no path to it')
                                en_without_path.append(escape_node)

                    # print(len(en_without_path), 'nodes without path')

                    # if city not in ['Utrecht', 'Rotterdam']:
                    assert set(exempt_ids_in_G).issubset(G_comb_coarsened.nodes())

                    cc = max(nx.weakly_connected_components(G), key=len)
                    for node in G_comb_coarsened.copy().nodes():
                        if node not in cc:
                            G_comb_coarsened.remove_node(node)

                    assert set(exempt_ids_in_G).issubset(G_comb_coarsened.nodes())


                    # if not set(exempt_ids_in_G).issubset(G_comb_coarsened.nodes()):
                    #     print('not all exempt ids in coarsened_G')


                    # for idx in exempt_ids_in_G:
                    #     if idx not in G_comb_coarsened.nodes():
                    #         print(idx)

                    print('number of nodes after adding detailed areas: ', len(G_comb_coarsened.nodes()))

                    # with open(f"networks/coarsened_network_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl", 'wb') as f:
                    #     pickle.dump(G_comb_coarsened, f)

                    ox.save_graphml(G_comb_coarsened, f"networks/panchamy_{weight_type}_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.graph.graphml")

                    # # Visualise the results
                    plot_network(weight_type, G_comb_coarsened, G, city, pruning, iterations, threshold, escape_nodes, fugitive_start, police_start)
