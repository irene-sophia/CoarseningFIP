import pickle
from time import gmtime, strftime
import osmnx as ox
import networkx as nx
import pandas as pd
from itertools import combinations
import itertools

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

def get_intercepted_routes(graph, route_data, results_positions, police_start):
    pi_nodes = {}

    for u, associated_node in enumerate(results_positions):

        # reken de reistijd naar de associated node uit
        travel_time_to_target = nx.shortest_path_length(graph,
                                                        source=police_start[u],
                                                        target=associated_node,
                                                        weight='travel_time',
                                                        method='bellman-ford')
        pi_nodes[u] = (associated_node, travel_time_to_target)

    result = set()
    for u_idx, pi_value in pi_nodes.items():  # for each police unit
        if pi_value[0] not in route_data:
            # print(pi_value)
            continue
        for fugitive_time in route_data[pi_value[0]]:
            if fugitive_time[1] >= (pi_value[1]):
                result.add(fugitive_time[0])

    return result


def route_convert(route_fugitive_labeled):
    """
    returns dict {node : [(route_idx, time_to_node), ...]
    """
    route_data = dict()
    for i_r, route in enumerate(route_fugitive_labeled):
        for time_at_node_fugitive, node_fugitive in route.items():
            if node_fugitive not in route_data:
                route_data[node_fugitive] = []
            route_data[node_fugitive].append((i_r, time_at_node_fugitive))

    return route_data


if __name__ == '__main__':

    results_cross_eval = []
    for weights in ['type', 'betweenness']:
        for city in ['Winterswijk', 'Manhattan', 'Utrecht', 'Amsterdam', 'Rotterdam']:
        # for city in ['Rotterdam']:
        #     for pruning in [0, 1]:
        #         for iterations in [1, 1000]:
        #             for threshold in [0, 1000]:
            for pruning in [0, 1]:
                for iterations in [1, 1000]:
                    for threshold in [0, 1000]:

                        filepath = f"networks/coarsened_network_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.graph.graphml"
                        G_c = ox.load_graphml(filepath=filepath)
                        filepath = f"networks/{city}.graph.graphml"
                        G = ox.load_graphml(filepath=filepath)

                        with open(f'networks/escape_nodes_{city}.pkl', 'rb') as f:
                            escape_nodes = pickle.load(f)
                        with open(f'networks/fugitive_start_{city}.pkl', 'rb') as f:
                            fugitive_start = pickle.load(f)

                        # get police routes
                        with open(f'networks/start_police_{city}.pkl', 'rb') as f:
                            police_start = pickle.load(f)
                        try:
                            if weights == 'type':
                                with open(f'results/results_positions_{city}_coarsened_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl', 'rb') as f:
                                    police_end_import = pickle.load(f)

                            elif weights == 'betweenness':
                                with open(
                                        f'results/betweenness_results_positions_{city}_coarsened_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl',
                                        'rb') as f:
                                    police_end_import = pickle.load(f)

                            with open(f'simulation/results/results_routes_sp_orig_{city}.pkl', 'rb') as f:
                                results_routes = pickle.load(f)

                            route_data = route_convert(results_routes)

                            results_routes = [list(route.values()) for route in results_routes]

                            best_combination_police_end = []
                            possible_end_nodes = []
                            attrs = nx.get_node_attributes(G_c, 'removed_nodes')
                            for u, pol_end in enumerate(police_end_import):
                                if pol_end in attrs.keys():
                                    possible_end_nodes_u_ = attrs[pol_end]
                                    possible_end_nodes_u = []
                                    for pos_node in possible_end_nodes_u_:
                                        if nx.has_path(G, police_start[u], pos_node):
                                            possible_end_nodes_u.append(pos_node)
                                    possible_end_nodes.append(possible_end_nodes_u)
                                else:
                                    assert pol_end in G_c.nodes()
                                    possible_end_nodes.append([pol_end])
                                    # print(pol_end)
                            combinations_possible_end_nodes = list(itertools.product(*possible_end_nodes))

                            best_pct_intercepted = -1
                            for police_end_ in combinations_possible_end_nodes:
                                routes_intercepted_ = get_intercepted_routes(G, route_data, police_end_, police_start)
                                print(len(routes_intercepted_))
                                if len(routes_intercepted_) > best_pct_intercepted:
                                    routes_intercepted = routes_intercepted_.copy()
                                    best_pct_intercepted = len(routes_intercepted_)
                                    police_end = police_end_

                            print(city, pruning, iterations, threshold, weights, 'intercepted: ',
                                  len(routes_intercepted) / len(results_routes))
                            results_cross_eval.append(len(routes_intercepted) / len(results_routes))

                            if len(routes_intercepted) == 0:
                                intercepted_routes = {r: 0 for r in range(len(results_routes))}
                            else:
                                intercepted_routes = {r: (1 if r in routes_intercepted else 0) for r in
                                                      range(len(results_routes))}

                            police_routes = [ox.shortest_path(G, police_start[u], police_end[u], weight='travel_time')
                                             for u, _ in
                                             enumerate(police_start)]

                            results_routes += police_routes

                            route_colors = ['tab:green' if val == 1 else 'tab:red' if val == 0 else ValueError for val
                                            in
                                            intercepted_routes.values()]
                            route_colors += ['tab:blue' for pol in police_routes]
                            route_alphas = [0.05 for fug in intercepted_routes]
                            route_alphas += [1 for pol in police_routes]
                            route_linewidths = [1 for fug in intercepted_routes]
                            route_linewidths += [2 for pol in police_routes]

                            node_size, node_color = draw_nodes(G, fugitive_start, escape_nodes, police_start,
                                                               police_end)
                            edge_colormap, edge_weightmap = draw_edges(G)
                            edge_weightmap = [0.3] * len(G.edges())

                            # fig, ax = ox.plot_graph_routes(G, results_routes,
                            #                                route_linewidths=route_linewidths, route_alpha=route_alphas,
                            #                                route_colors=route_colors,
                            #                                edge_linewidth=edge_weightmap, edge_color=edge_colormap,
                            #                                node_color=node_color, node_size=node_size, node_zorder=2,
                            #                                bgcolor="white",
                            #                                orig_dest_size=30,
                            #                                show=False,
                            #                                # orig_dest_node_color=['tab:orange', 'tab:red']*len(results_routes),
                            #                                )
                            #
                            # fig.savefig(f'figs/crossevaluation_{city}.png', bbox_inches='tight', dpi=300)
                        except:
                            print(city, pruning, iterations, threshold, weights, 'no result')
                            results_cross_eval.append(0)

    data = pd.read_excel('./results/node reduction.xlsx')
    data['score, coarsened eval opt'] = results_cross_eval
    data['% degradation opt'] = (-1*(1-(data['score, coarsened eval opt']/ data['score, orig'])))*100
    data.to_excel('./results/node reduction.xlsx', index=False)
