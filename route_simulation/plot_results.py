import pickle

import osmnx as ox
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def draw_edges(graph):
    edges_fugitive = []

    # for i_r, route_time in enumerate(fugitive_routes):
    #     route = list(route_time.values())
    #     for i, node in enumerate(route):
    #         if i ==0:
    #             continue
    #         else:
    #             edges_fugitive1 = [(route[i], route[i-1])]
    #             edges_fugitive2 = [(route[i-1], route[i])]
    #             edges_fugitive.extend(tuple(edges_fugitive1))
    #             edges_fugitive.extend(tuple(edges_fugitive2))

    edge_colormap = ['lightgray'] * len(graph.edges())
    edge_weightmap = [1] * len(graph.edges())
    for index, edge in enumerate(graph.edges()):
        if edge in edges_fugitive:
            edge_colormap[index] = 'tab:orange'
            edge_weightmap[index] = 2

    return edge_colormap, edge_weightmap


def draw_nodes(G, fugitive_start, escape_nodes):
    node_size = []
    node_color = []
    for node in G.nodes:
        if node in escape_nodes:
            node_size.append(30)
            node_color.append('tab:red')
        elif node == fugitive_start:
            node_size.append(40)
            node_color.append('tab:orange')
        else:
            node_size.append(0)
            node_color.append('lightgray')
    return node_size, node_color


def show_graph(G, escape_nodes, fugitive_start, save=False):
    # filepath=f"graphs/FLEE/Graph_FLEE.graph.graphml"
    # G = ox.load_graphml(filepath=filepath)

    edge_colormap, edge_weightmap = draw_edges(G)
    node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes)

    fig, ax = ox.plot_graph(
        G, bgcolor="white", node_color=node_color, node_size=node_size, edge_linewidth=edge_weightmap,
        edge_color=edge_colormap,
    )
    if save:
        ax.savefig(f'graphs/{city}.png')


def plot_routes(city, weight_type, pruning, iterations, threshold):
    filepath = f"../../data/networks/panchamy_{weight_type}_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.graph.graphml"
    G = ox.load_graphml(filepath=filepath)

    with open(f'../../data/networks/escape_nodes_{city}.pkl', 'rb') as f:
        escape_nodes = pickle.load(f)

    with open(f'../../data/networks/fugitive_start_{city}.pkl', 'rb') as f:
        fugitive_start = pickle.load(f)

    # if city != 'Winterswijk':
    #     with open(f'data/cameras_{city}.pkl', 'rb') as f:
    #         cameras = pickle.load(f)
    # elif city == 'Winterswijk':
    #     cameras = []

    cameras = []

    # nx.draw_networkx_edges(G,edgelist=path_edges,edge_color='r',width=10)
    node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes)
    edge_colormap, edge_weightmap = draw_edges(G)

    with open(f'results/results_routes_sp_coarsened_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl', 'rb') as f:
        results_routes = pickle.load(f)

    results_routes = [list(route.values()) for route in results_routes]

    fig, ax = ox.plot_graph_routes(G, results_routes,
                                   route_linewidths=1, route_alphas=[0.05]*len(results_routes),
                                   edge_linewidth=edge_weightmap, edge_color=edge_colormap,
                                   node_color=node_color, node_size=node_size,
                                   bgcolor="white",
                                   orig_dest_size=30,
                                   show=False,
                                   # orig_dest_node_color=['tab:orange', 'tab:red']*len(results_routes),
                                   )

    fig.savefig(f'figs/routes_{weight_type}_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.png', bbox_inches='tight', dpi=300)

    # generate shortest paths for ABM
    # route_fugitive = []
    # while len(route_fugitive) < 10:
    #     for escape_node in escape_nodes:
    #         try:
    #             path = nx.shortest_path(G, fugitive_start, escape_node, 'travel_time')
    #             # [escape_routes.append(path) for path in nx.all_simple_paths(G, fugitive_start, escape_node)]
    #             route_fugitive.append(path)
    #         except:
    #             continue
    #
    # with open(f'data/escape_routes_{city}.pkl', 'wb') as f:
    #     pickle.dump(route_fugitive, f)


if __name__ == '__main__':
    city = 'Utrecht'
    weight_type = 'betweenness'

    # for city in ['Utrecht']:
    for city in ['Winterswijk', 'Manhattan', 'Utrecht']:
        for pruning in [0, 1]:
            for iterations in [1, 1000]:
                for threshold in [0, 1000]:
                    plot_routes(city, weight_type, pruning, iterations, threshold)
