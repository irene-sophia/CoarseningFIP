import pickle
from time import gmtime, strftime
import osmnx as ox
import logging
from datetime import datetime
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def draw_edges(graph):
    edges_fugitive = []

    edge_colormap = ['lightgray'] * len(graph.edges())
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

        # elif node in police_start:
        #     node_size.append(60)
        #     node_color.append('tab:blue')
        if node in police_start:
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


def plot_routes(city, graph_type):
    if graph_type == 'coarsened':
        filepath = f"networks/coarsened_network_{city}.graph.graphml"
    elif graph_type == 'orig':
        filepath = f"networks/{city}.graph.graphml"

    G = ox.load_graphml(filepath=filepath)

    with open(f'networks/escape_nodes_{city}.pkl', 'rb') as f:
        escape_nodes = pickle.load(f)
    with open(f'networks/fugitive_start_{city}.pkl', 'rb') as f:
        fugitive_start = pickle.load(f)

    # get police routes
    with open(f'networks/start_police_{city}.pkl', 'rb') as f:
        police_start = pickle.load(f)

    # with open(f'results/optimization/results_positions_{mode}_{city}_jitter{jitter}.pkl', 'rb') as f:
    #     police_end = pickle.load(f)

    with open(f'simulation/results/results_routes_sp_{graph_type}_{city}.pkl', 'rb') as f:
        results_routes = pickle.load(f)
    results_routes = [list(route.values()) for route in results_routes]
    # results_routes += police_routes

    # if mode == 'hot+cool':
    #     with open(f'results/optimization/results_intercepted_routes_{mode}_{city}.pkl', 'rb') as f:
    #         intercepted_routes = pickle.load(f)
    # else:
    #     with open(f'results/optimization/results_intercepted_routes_{mode}_{city}.pkl', 'rb') as f:
    #         intercepted_routes = pickle.load(f)

    # route_colors = ['tab:green' if val == 1 else 'tab:red' if val == 0 else ValueError for val in intercepted_routes.values()]
    # route_colors += ['tab:blue' for pol in police_routes]
    # route_alphas = [0.05 for fug in intercepted_routes]
    # # route_alphas += [1 for pol in police_routes]
    # route_linewidths = [1 for fug in intercepted_routes]
    # # route_linewidths += [2 for pol in police_routes]

    route_alphas = [0.05 for fug in results_routes]
    route_linewidths = [1 for fug in results_routes]
    route_colors = ['tab:red'] * len(results_routes)

    # nx.draw_networkx_edges(G,edgelist=path_edges,edge_color='r',width=10)
    node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes, police_start, [])
    edge_colormap, edge_weightmap = draw_edges(G)
    edge_weightmap = [0.3] * len(G.edges())


    fig, ax = ox.plot_graph_routes(G, results_routes,
                                   route_linewidths=route_linewidths, route_alphas=0.05, route_colors=route_colors,
                                   edge_linewidth=edge_weightmap, edge_color=edge_colormap,
                                   node_color=node_color, node_size=node_size, node_zorder=2,
                                   bgcolor="white",
                                   orig_dest_size=30,
                                   show=False,
                                   # orig_dest_node_color=['tab:orange', 'tab:red']*len(results_routes),
                                   )

    fig.savefig(f'figs/routes_{city}_{graph_type}.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':

    graph_type = 'orig'  # orig / coarsened
    # for city in ['Winterswijk']:
    # for city in ['Manhattan', 'Winterswijk', 'Utrecht']:
    for city in ['Amsterdam']:
        for graph_type in ['orig']:
            plot_routes(city, graph_type)
            print(datetime.now().strftime("%H:%M:%S"), 'done: ', graph_type, city)
