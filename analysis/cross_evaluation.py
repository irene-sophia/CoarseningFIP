import pickle
from time import gmtime, strftime
import osmnx as ox
import networkx as nx
import pandas as pd
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
        try:
            # reken de reistijd naar de associated node uit
            travel_time_to_target = nx.shortest_path_length(graph,
                                                            source=police_start[u],
                                                            target=associated_node,
                                                            weight='travel_time',
                                                            method='bellman-ford')
        except:
            travel_time_to_target = 424242
            print('cannot reach ', associated_node, 'from ', police_start[u])

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

    approach = 'onthefly'

    max_iter_dict = {
        'Winterswijk': 5,
        'Manhattan': 1,
        'Utrecht': 6,
        'Amsterdam': 6,
        'Rotterdam': 7
    }

    SSRs = ['without_SSR', 'with_SSR']
    cities = ['Winterswijk', 'Manhattan', 'Utrecht', 'Amsterdam', 'Rotterdam']
    weights = ['type', 'betweenness']
    prunings = [0, 1]
    iterationss = [1, 1000]
    thresholds = [0, 1000]
    tolerances = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    seeds = range(10)

    if approach == 'original':
        iterator = itertools.product(SSRs, cities, seeds)
    elif approach == 'pruning':
        iterator = itertools.product(SSRs, cities, seeds)
    elif approach == 'onthefly':
        iterator = itertools.product(SSRs, cities, seeds)
    elif approach == 'consolidated':
        iterator = itertools.product(SSRs, cities, tolerances, seeds)
    elif approach == 'panchamy':
        iterator = itertools.product(SSRs, weights, cities, prunings, iterationss, thresholds, seeds)

    results_cross_eval = {}

    for SSR, city, seed in iterator:  # orig / onthefly
    # for SSR, city, tolerance, seed in iterator:  # consolidated
    # for SSR, weight, city, pruning, iterations, threshold, seed in iterator:  # panchamy
    # for SSR, city, seed in iterator:  # pruning
    #     for iterations_pruning in range(max_iter_dict[city]):# pruning
        filepath = f"../data/networks/{city}.graph.graphml"
        G = ox.load_graphml(filepath=filepath)

        with open(f'../data/networks/escape_nodes_{city}.pkl', 'rb') as f:
            escape_nodes = pickle.load(f)
        with open(f'../data/networks/fugitive_start_{city}.pkl', 'rb') as f:
            fugitive_start = pickle.load(f)

        # get police routes
        with open(f'../data/networks/start_police_{city}.pkl', 'rb') as f:
            police_start = pickle.load(f)

        try:
            if approach == 'original':
                with open(f'../HPC_results/{SSR}/{approach}/results_positions_{approach}_{city}_seed{seed}.pkl', 'rb') as f:
                    police_end = pickle.load(f)
            elif approach == 'pruning':
                with open(f'../HPC_results/{SSR}/{approach}/results_positions_{approach}_{city}_iter{iterations_pruning}_seed{seed}.pkl', 'rb') as f:
                    police_end = pickle.load(f)
            elif approach == 'onthefly':
                with open(f'../HPC_results/{SSR}/{approach}/results_positions_{approach}_{city}_seed{seed}.pkl', 'rb') as f:
                    police_end = pickle.load(f)
            elif approach == 'consolidated':
                with open(f'../HPC_results/{SSR}/{approach}/results_positions_{approach}_{city}_{tolerance}_seed{seed}.pkl', 'rb') as f:
                    police_end = pickle.load(f)
            elif approach == 'panchamy':
                with open(f'../HPC_results/{SSR}/{approach}/results_positions_{approach}_{city}_{weight}_pruning{pruning}_iter{iterations}_threshold{threshold}_seed{seed}.pkl', 'rb') as f:
                    police_end = pickle.load(f)

            with open(f'../data/routes/orig/results_routes_sp_orig_{city}.pkl', 'rb') as f:
                results_routes = pickle.load(f)

            police_routes = []
            for u, _ in enumerate(police_start):
                try:
                    police_routes.append(ox.shortest_path(G, police_start[u], police_end[u], weight='travel_time'))
                except:
                    police_routes.append([police_start])
            # police_routes = [ox.shortest_path(G, police_start[u], police_end[u], weight='travel_time')
            #                  for u, _ in
            #                  enumerate(police_start)]

            route_data = route_convert(results_routes)
            results_routes = [list(route.values()) for route in results_routes]
            routes_intercepted = get_intercepted_routes(G, route_data, police_end, police_start)

            if approach == 'original':
                print(SSR, city, seed, 'intercepted: ',
                      len(routes_intercepted) / len(results_routes))
                results_cross_eval[SSR, city, seed] = len(routes_intercepted) / len(results_routes)
            elif approach == 'pruning':
                print(SSR, city, iterations_pruning, seed, 'intercepted: ',
                      len(routes_intercepted) / len(results_routes))
                results_cross_eval[SSR, city, iterations_pruning, seed] = len(routes_intercepted) / len(results_routes)
            elif approach == 'onthefly':
                print(SSR, city, seed, 'intercepted: ',
                      len(routes_intercepted) / len(results_routes))
                results_cross_eval[SSR, city, seed] = len(routes_intercepted) / len(results_routes)
            elif approach == 'consolidated':
                print(SSR, city, tolerance, seed, 'intercepted: ',
                      len(routes_intercepted) / len(results_routes))
                results_cross_eval[SSR, city, tolerance, seed] = len(routes_intercepted) / len(results_routes)
            elif approach == 'panchamy':
                print(SSR, weight, city, pruning, iterations, threshold, seed, 'intercepted: ',
                      len(routes_intercepted) / len(results_routes))
                results_cross_eval[SSR, weight, city, pruning, iterations, threshold, seed] = len(routes_intercepted) / len(results_routes)
        except:
            pass

    print(results_cross_eval)
    with open(f'./cleaned_data/crosseval_{approach}.pkl', 'wb') as f:
        pickle.dump(results_cross_eval, f)

