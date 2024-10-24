import networkx as nx
import numpy as np
import string
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
import pickle
import ast


def node_colors(G, escape_nodes, fugitive_start, police_start):
    node_size = []
    node_color = []
    for u, data in G.nodes(data=True):
        if data['osmid_original'] == fugitive_start:
            node_size.append(40)
            node_color.append('tab:orange')
        elif data['osmid_original'] in police_start:
            node_size.append(40)
            node_color.append('tab:blue')
        elif data['osmid_original'] in escape_nodes:
            node_size.append(40)
            node_color.append('tab:red')
        # if u == fugitive_start:
        #     node_size.append(40)
        #     node_color.append('tab:orange')
        # elif u in police_start:
        #     node_size.append(40)
        #     node_color.append('tab:blue')
        # elif u in escape_nodes:
        #     node_size.append(40)
        #     node_color.append('tab:red')

        else:
            node_size.append(0)
            node_color.append('lightgray')

    return node_size, node_color


def edge_colors(G):
    edge_color = ['lightgray'] * len(G.edges())
    edge_size = [1] * len(G.edges())
    # for index, edge in enumerate(G.edges()):
    #     if edge in edges_fugitive:
    #         edge_color[index] = 'tab:orange'
    #         edge_size[index] = 2

    return edge_size, edge_color


if __name__ == '__main__':
    city = 'Amsterdam'
    num_nodes = []
    for city in ['Manhattan', 'Winterswijk', 'Utrecht', 'Amsterdam', 'Rotterdam']:
        filepath = f"../data/networks/{city}.graph.graphml"
        # ox.save_graph_geopackage(G, filepath=filepath)
        G = ox.load_graphml(filepath=filepath)

        with open(f'../data/networks/escape_nodes_{city}.pkl', 'rb') as f:
            escape_nodes_ = pickle.load(f)

        G.graph['crs'] = 4326

        G_con = ox.project_graph(G)

        # x_dict = {}
        # y_dict = {}
        # for node, data in G_con.nodes(data=True):
        #     x_dict[data['x']] = data['lon']
        #     y_dict[data['y']] = data['lat']

        # nodes_xy = {}
        # for node, data in G_con.nodes(data=True):
        #     nodes_xy[node] = {'x': data['lon'], 'y': data['lat']}
        # nx.set_node_attributes(G_con, nodes_xy)

        for tolerance in [50, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            G_pruned = ox.simplification.consolidate_intersections(G_con, tolerance=tolerance, rebuild_graph=True, dead_ends=False)

            # filepath = f"results/networks/consolidated_network_{city}_{tolerance}.graph.graphml"
            # G_pruned = ox.load_graphml(filepath=filepath)
            print(len(G_pruned.nodes()))
            num_nodes.append(len(G_pruned.nodes()))

            # with open(f'../data/networks/fugitive_start_{city}.pkl', 'rb') as f:
            #     fugitive_start = pickle.load(f)
            # escape_nodes = []
            # for node in escape_nodes_:
            #     if node in G.nodes():
            #         if nx.has_path(G, source=fugitive_start, target=node):
            #             escape_nodes.append(node)
            #
            # if fugitive_start not in G_pruned.nodes:
            #     for node, data in G_pruned.nodes(data=True):
            #         if isinstance(data['osmid_original'], int):
            #             if fugitive_start == data['osmid_original']:
            #                 print(fugitive_start)
            #                 fugitive_start = node
            #                 print(fugitive_start)
            #                 break
            #         elif isinstance(data['osmid_original'], str):
            #             if isinstance(ast.literal_eval(data['osmid_original']), int):
            #                 if fugitive_start == data['osmid_original']:
            #                     print(fugitive_start)
            #                     fugitive_start = node
            #                     print(fugitive_start)
            #                     break
            #             elif isinstance(data['osmid_original'], list):
            #                 if fugitive_start in ast.literal_eval(data['osmid_original']):
            #                     print(fugitive_start)
            #                     fugitive_start = node
            #                     print(fugitive_start)
            #                     break
            #         elif isinstance(data['osmid_original'], list):
            #             if fugitive_start in data['osmid_original']:
            #                 print(fugitive_start)
            #                 fugitive_start = node
            #                 print(fugitive_start)
            #                 break
            #
            # with open(f'../data/networks/start_police_{city}.pkl', 'rb') as f:
            #     start_police = pickle.load(f)
            # for i, pol in enumerate(start_police):
            #     if pol not in G_pruned.nodes:
            #         # print(pol, 'not in nodes')
            #         for node, data in G_pruned.nodes(data=True):
            #             if isinstance(data['osmid_original'], int):
            #                 if pol == data['osmid_original']:
            #                     start_police[i] = node
            #                     print(i, pol, node)
            #             elif isinstance(data['osmid_original'], list):
            #                 if pol in data['osmid_original']:
            #                     start_police[i] = node
            #                     print(i, pol, node)
            #             elif isinstance(data['osmid_original'], str):
            #                 if isinstance(ast.literal_eval(data['osmid_original']), int):
            #                     if pol == ast.literal_eval(data['osmid_original']):
            #                         start_police[i] = node
            #                         print(i, pol, node)
            #                 elif isinstance(ast.literal_eval(data['osmid_original']), list):
            #                     if fugitive_start in ast.literal_eval(data['osmid_original']):
            #                         start_police[i] = node
            #                         print(i, pol, node)
            #
            # if city == 'Amsterdam':
            #     from shapely import LineString
            #
            #     for u, v, data in G_pruned.edges(data=True):
            #         if type((data['geometry'])) != LineString:
            #             try:
            #                 data['geometry'] = data['geometry'][data['u_original'], data['v_original'], 0]
            #                 nx.set_edge_attributes(G_pruned, {(u, v, 0): data})
            #             except:
            #                 del data['geometry']
            #                 print(data)
            #                 nx.set_edge_attributes(G_pruned, data)
            #
            #     for u, v, data in G_pruned.edges(data=True):
            #         if type((data['geometry'])) != LineString:
            #             del data['geometry']
            #             print(data)
            #             nx.set_edge_attributes(G_pruned, {(u, v, 0): data})
            #
            # node_size, node_color = node_colors(G_pruned, escape_nodes, fugitive_start, start_police)
            # edge_size, edge_color = edge_colors(G_pruned)
            # fig, ax = ox.plot_graph(G_pruned,
            #               bgcolor="white", node_color=node_color, node_size=node_size,
            #               edge_linewidth=edge_size, edge_color=edge_color,
            #               # show=True, save=True, filepath=f'figs/{city}_pruned_{tolerance}.png'
            #               show=False
            #               )
            # fig.savefig(f'./figs/{city}_pruned_{tolerance}.png', dpi=300, bbox_inches='tight')
            # plt.close(fig)
            #
            # pruned_ids_dict = {}
            # for node, data in G_pruned.nodes(data=True):
            #     if isinstance(data['osmid_original'], int):
            #         pruned_ids_dict[node] = int(data['osmid_original'])
            #     elif isinstance(data['osmid_original'], str):
            #         pruned_ids_dict[node] = ast.literal_eval(data['osmid_original'])[0]
            #     elif isinstance(data['osmid_original'], float):
            #         pruned_ids_dict[node] = int(data['osmid_original'])
            #     elif isinstance(data['osmid_original'], list):
            #         print(node, 'hey')
            #         pruned_ids_dict[node] = int(data['osmid_original'][0])
            # nx.relabel_nodes(G_pruned, pruned_ids_dict, copy=False)
            #
            # ox.save_graphml(G_pruned, f"results/networks/consolidated_network_{city}_{tolerance}.graph.graphml")
            # print('done: ', city, tolerance)

    # print(num_nodes)
    # data = pd.read_excel('./node reduction.xlsx')
    # data['number of nodes coarsened graph'] = num_nodes
    # data.to_excel('./node reduction.xlsx', index=False)