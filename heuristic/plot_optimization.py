import pickle
from time import gmtime, strftime
import osmnx as ox
import logging
from datetime import datetime
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from cut_graph_to_boundaries import cut_graph

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


def plot_routes(city, graph_type, pruning, iterations, threshold):
    if graph_type == 'coarsened':
        filepath = f"networks/coarsened_network_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.graph.graphml"
    elif graph_type == 'orig':
        filepath = f"networks/{city}.graph.graphml"

    G = ox.load_graphml(filepath=filepath)
    G = cut_graph(G, city)

    with open(f'networks/escape_nodes_{city}.pkl', 'rb') as f:
        escape_nodes = pickle.load(f)
    with open(f'networks/fugitive_start_{city}.pkl', 'rb') as f:
        fugitive_start = pickle.load(f)

    if graph_type == 'orig':
        with open(f'simulation/results/results_routes_sp_{graph_type}_{city}.pkl', 'rb') as f:
            results_routes = pickle.load(f)
        with open(f'results/results_intercepted_routes_{city}_{graph_type}.pkl', 'rb') as f:
            intercepted_routes_set = pickle.load(f)
        with open(f'results/results_positions_{city}_{graph_type}.pkl', 'rb') as f:
            police_end = pickle.load(f)
    elif graph_type == 'coarsened':
        with open(f'simulation/results/results_routes_sp_{graph_type}_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl', 'rb') as f:
            results_routes = pickle.load(f)
        with open(f'results/results_intercepted_routes_{city}_{graph_type}_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl', 'rb') as f:
            intercepted_routes_set = pickle.load(f)
        with open(f'results/results_positions_{city}_{graph_type}_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl', 'rb') as f:
            police_end = pickle.load(f)
    results_routes = [list(route.values()) for route in results_routes]

    if len(intercepted_routes_set) == 0:
        intercepted_routes = {r: 0 for r in range(len(results_routes))}
    else:
        intercepted_routes = {r: (1 if r in intercepted_routes_set else 0) for r in range(len(results_routes))}

    # get police routes
    with open(f'networks/start_police_{city}.pkl', 'rb') as f:
        police_start = pickle.load(f)
    police_routes = [ox.shortest_path(G, police_start[u], police_end[u], weight='travel_time') for u, _ in enumerate(police_start)]
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
    node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes, police_start, police_end)
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
    if graph_type == 'orig':
        fig.savefig(f'figs/opstelpos_{city}_{graph_type}.png', bbox_inches='tight', dpi=300)
    elif graph_type == 'coarsened':
        fig.savefig(f'figs/opstelpos_{city}_{graph_type}_pruning{pruning}_iter{iterations}_threshold{threshold}.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    graph_type = 'orig'
    # for city in ['Manhattan', 'Utrecht', 'Winterswijk']:  #
    for city in ['Rotterdam']:
        plot_routes(city, graph_type, 0, 0, 0)
        print(datetime.now().strftime("%H:%M:%S"), 'done: ', graph_type, city)

    graph_type = 'coarsened'
    for city in ['Rotterdam']:
    # for city in ['Winterswijk', 'Utrecht', 'Manhattan']:
        for pruning in [0, 1]:
            for iterations in [1, 1000]:
                for threshold in [0, 1000]:
                    plot_routes(city, graph_type, pruning, iterations, threshold)
                    print(datetime.now().strftime("%H:%M:%S"), 'done: ', graph_type, city, pruning, iterations, threshold)
