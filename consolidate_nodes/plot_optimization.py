import pickle
from time import gmtime, strftime
import osmnx as ox
import logging
from datetime import datetime
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import ast


def draw_edges(graph):
    edges_fugitive = []

    edge_colormap = ['silver'] * len(graph.edges())
    edge_weightmap = [1] * len(graph.edges())
    for index, edge in enumerate(graph.edges()):
        if edge in edges_fugitive:
            edge_colormap[index] = 'tab:orange'
            edge_weightmap[index] = 2

    return edge_colormap, edge_weightmap


def draw_nodes(G, fugitive_start, escape_nodes, police_start, police_end):
    node_size = []
    node_color = []
    for node in G.nodes:
        # if node in police_end:
        #     node_size.append(120)
        #     node_color.append('tab:blue')

        if node in police_end:
            node_size.append(60)
            node_color.append('tab:blue')
        elif node in police_start:
            node_size.append(60)
            node_color.append('#51a9ff')
        elif node == fugitive_start:
            node_size.append(40)
            node_color.append('tab:orange')
        elif node in escape_nodes:
            node_size.append(20)
            node_color.append('tab:red')
        else:
            node_size.append(0)
            node_color.append('lightgray')
    return node_size, node_color


def plot_routes(city, tolerance):
    filepath = f"results/networks/consolidated_network_{city}_{tolerance}.graph.graphml"
    G = ox.load_graphml(filepath=filepath)

    with open(f'../networks/escape_nodes_{city}.pkl', 'rb') as f:
        escape_nodes = pickle.load(f)
    with open(f'../networks/fugitive_start_{city}.pkl', 'rb') as f:
        fugitive_start = pickle.load(f)
    if fugitive_start not in G.nodes:
        for node, data in G.nodes(data=True):
            if isinstance(data['osmid_original'], int):
                if fugitive_start == data['osmid_original']:
                    print(fugitive_start)
                    fugitive_start = node
                    print(fugitive_start)
                    break
            elif isinstance(data['osmid_original'], str):
                if isinstance(ast.literal_eval(data['osmid_original']), int):
                    if fugitive_start == data['osmid_original']:
                        print(fugitive_start)
                        fugitive_start = node
                        print(fugitive_start)
                        break
                elif isinstance(data['osmid_original'], list):
                    if fugitive_start in ast.literal_eval(data['osmid_original']):
                        print(fugitive_start)
                        fugitive_start = node
                        print(fugitive_start)
                        break
            elif isinstance(data['osmid_original'], list):
                if fugitive_start in data['osmid_original']:
                    print(fugitive_start)
                    fugitive_start = node
                    print(fugitive_start)
                    break

    with open(f'../networks/start_police_{city}.pkl', 'rb') as f:
        start_police = pickle.load(f)
    for i, pol in enumerate(start_police):
        if pol not in G.nodes:
            # print(pol, 'not in nodes')
            for node, data in G.nodes(data=True):
                if isinstance(data['osmid_original'], int):
                    if pol == data['osmid_original']:
                        start_police[i] = node
                        print(i, pol, node)
                elif isinstance(data['osmid_original'], list):
                    if pol in data['osmid_original']:
                        start_police[i] = node
                        print(i, pol, node)
                elif isinstance(data['osmid_original'], str):
                    if isinstance(ast.literal_eval(data['osmid_original']), int):
                        if pol == ast.literal_eval(data['osmid_original']):
                            start_police[i] = node
                            print(i, pol, node)
                    elif isinstance(ast.literal_eval(data['osmid_original']), list):
                        if fugitive_start in ast.literal_eval(data['osmid_original']):
                            start_police[i] = node
                            print(i, pol, node)

    with open(f'results/routes/results_routes_sp_consolidated_{city}_{tolerance}.pkl', 'rb') as f:
        results_routes = pickle.load(f)
    with open(f'results/optimization/consolidated_results_intercepted_routes_{city}_{tolerance}.pkl', 'rb') as f:
        intercepted_routes_set = pickle.load(f)
    with open(f'results/optimization/consolidated_results_positions_{city}_{tolerance}.pkl', 'rb') as f:
        police_end = pickle.load(f)
    results_routes = [list(route.values()) for route in results_routes]

    if len(intercepted_routes_set) == 0:
        intercepted_routes = {r: 0 for r in range(len(results_routes))}
    else:
        intercepted_routes = {r: (1 if r in intercepted_routes_set else 0) for r in range(len(results_routes))}

    # get police routes
    police_routes = [ox.shortest_path(G, start_police[u], police_end[u], weight='travel_time') for u, _ in enumerate(start_police)]
    results_routes += police_routes

    route_colors = ['tab:green' if val == 1 else 'tab:red' if val == 0 else ValueError for val in intercepted_routes.values()]
    route_colors += ['tab:blue' for pol in police_routes]
    route_alphas = [0.05 for fug in intercepted_routes]
    route_alphas += [1 for pol in police_routes]
    route_linewidths = [1 for fug in intercepted_routes]
    route_linewidths += [2 for pol in police_routes]

    # route_alphas = [0.05 for fug in intercepted_routes]
    # route_linewidths = [1 for fug in intercepted_routes]
    # route_colors = ['tab:red'] * len(results_routes)

    # nx.draw_networkx_edges(G,edgelist=path_edges,edge_color='r',width=10)
    node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes, start_police, police_end)
    edge_colormap, edge_weightmap = draw_edges(G)
    edge_weightmap = [0.3] * len(G.edges())


    fig, ax = ox.plot_graph_routes(G, results_routes,
                                   route_linewidths=route_linewidths, route_alphas=route_alphas, route_colors=route_colors,
                                   edge_linewidth=edge_weightmap, edge_color=edge_colormap,
                                   node_color=node_color, node_size=node_size, node_zorder=2,
                                   bgcolor="white",
                                   orig_dest_size=30,
                                   show=False,
                                   # orig_dest_node_color=['tab:orange', 'tab:red']*len(results_routes),
                                   )

    fig.savefig(f'figs/optimization/opstelpos_{city}_{tolerance}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    for city in ['Winterswijk']:
    # for city in ['Winterswijk', 'Utrecht', 'Manhattan']:
        for tolerance in [1]:
            plot_routes(city, tolerance)
            print(datetime.now().strftime("%H:%M:%S"), 'done: ', city, tolerance)
