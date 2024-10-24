import pickle
import time
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd

from ema_workbench import MultiprocessingEvaluator, SequentialEvaluator
from ema_workbench import RealParameter, ScalarOutcome, Constant, Model
from ema_workbench.em_framework.optimization import ArchiveLogger, SingleObjectiveBorgWithArchive

from panchamy.optimization.sort_and_filter import sort_and_filter_pol_fug_city as sort_and_filter_nodes
from panchamy.optimization.sort_and_filter import sort_nodes as sort_nodes
from panchamy.cut_graph_to_boundaries import cut_graph
from network_contruction import construct_network

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


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

def FIP_func(
        graph=None,
        labels=None,
        labels_perunit_sorted_inv=None,
        police_start=None,
        route_data=None,
        t_max=None,
        **kwargs,
):
    pi_nodes = {}

    for u, value in enumerate(kwargs.values()):
        associated_node = labels_perunit_sorted_inv[u][int(np.floor(value))]
        # reken hier 1 keer de reistijd naar de associated node uit ipv die hele matrix
        if nx.has_path(graph, source=police_start[u], target=associated_node):
            travel_time_to_target = nx.shortest_path_length(graph,
                                                          source=police_start[u],
                                                          target=associated_node,
                                                          weight='travel_time',
                                                          method='bellman-ford')
        else:
            travel_time_to_target = 424242
            # print(police_start[u], associated_node, travel_time_to_target)
        # associated_node = labels[associated_node]
        pi_nodes[u] = (associated_node, travel_time_to_target)

    result = set()
    for u_idx, pi_value in pi_nodes.items():  # for each police unit
        if pi_value[0] not in route_data:
            # print(pi_value)
            continue
        for fugitive_time in route_data[pi_value[0]]:
            if fugitive_time[1] >= (pi_value[1]):
                result.add(fugitive_time[0])
    # return {'num_intercepted': float(len(result))}
    # print(float(len(result)))
    return [float(len(result))]


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


def optimize(graph, police_start, upper_bounds, num_units, route_data, t_max,
             labels, labels_perunit_inv_sorted, SSR):
    if SSR:
        folder = 'with_SSR'
    else:
        folder = 'without_SSR'

    model = Model("FIPEMA", function=FIP_func)

    model.levers = [RealParameter(f"pi_{u}", 0, upper_bounds[u]) for u in range(num_units)]

    model.constants = model.constants = [
        Constant("route_data", route_data),
        Constant("t_max", t_max),
        # Constant("tau_uv", tau_uv),
        Constant("labels", labels),
        Constant("labels_perunit_sorted_inv", labels_perunit_inv_sorted),
        # Constant("time_step", time_step),
        Constant("graph", graph),
        Constant("police_start", police_start),
    ]

    model.outcomes = [
        ScalarOutcome("num_intercepted", kind=ScalarOutcome.MAXIMIZE)
    ]

    highest_perf = 0
    # with MultiprocessingEvaluator(model, n_processes=5) as evaluator:
    with SequentialEvaluator(model) as evaluator:
        for _ in range(1):  # TODO: change to 5
            convergence_metrics = [
                ArchiveLogger(
                    f"./results/{folder}/onthefly/",
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes if o.kind != o.INFO],
                    base_filename=f"archives_onthefly_{city}_seed{_}.tar.gz"
                ),
            ]

            result = evaluator.optimize(
                algorithm=SingleObjectiveBorgWithArchive,
                nfe=100,  # TODO: change to 100 000
                searchover="levers",
                convergence=convergence_metrics,
                convergence_freq=100
            )

            result = result.iloc[0]
            print(_, result['num_intercepted'])
            if result['num_intercepted'] >= highest_perf:
                results = result
                highest_perf = result['num_intercepted']

            convergence = ArchiveLogger.load_archives(
                f"./results/{folder}/onthefly/archives_onthefly_{city}_seed{_}.tar.gz")
            convergence_df = pd.DataFrame()
            for nfe, archive in convergence.items():
                archive['nfe'] = nfe
                convergence_df = pd.concat([convergence_df, archive])
            convergence_df.to_csv(
                f'./results/{folder}/onthefly/convergence_onthefly_{city}_seed{_}.csv')

            results_positions_ = []
            results_positions_labeled_ = []
            for u, start in enumerate(police_start):
                results_positions_labeled_.append(result[f'pi_{u}'])
                results_positions_.append(labels_perunit_inv_sorted[u][int(np.floor(result[f'pi_{u}']))])

            routes_intercepted_ = get_intercepted_routes(graph, route_data, results_positions_, police_start)
            with open(
                    f'./results/{folder}/onthefly/results_intercepted_routes_onthefly_{city}_seed{_}.pkl',
                    'wb') as f:
                pickle.dump(routes_intercepted_, f)

            with open(
                    f'./results/{folder}/onthefly/results_positions_onthefly_{city}_seed{_}.pkl',
                    'wb') as f:
                pickle.dump(results_positions_, f)

    results_positions = []
    results_positions_labeled = []
    for u, start in enumerate(police_start):
        results_positions_labeled.append(results[f'pi_{u}'])
        results_positions.append(labels_perunit_inv_sorted[u][int(np.floor(results[f'pi_{u}']))])
    # print(results_positions)

    routes_intercepted = get_intercepted_routes(graph, route_data, results_positions, police_start)
    # print(routes_intercepted)

    return results, routes_intercepted, results_positions


if __name__ == '__main__':
    t_max = 1800
    SSR = 0

    # for city in ['Amsterdam']:
    for city in ['Winterswijk', 'Utrecht', 'Manhattan', 'Amsterdam', 'Rotterdam']:
        print(city)
        G_orig = ox.load_graphml(filepath=f"../data/networks/{city}.graph.graphml")
        # if city != 'Amsterdam':
        #     G_orig = cut_graph(G_orig, city)

        # import escape nodes
        with open(f'../data/networks/escape_nodes_{city}.pkl', 'rb') as f:
            escape_nodes = pickle.load(f)

        # import start fugitive
        with open(f'../data/networks/fugitive_start_{city}.pkl', 'rb') as f:
            fugitive_start = pickle.load(f)

        # import starts police
        with open(f'../data/networks/start_police_{city}.pkl', 'rb') as f:
            start_police = pickle.load(f)

        # import delays police
        with open(f'../data/networks/delays_police_{city}.pkl', 'rb') as f:
            delays_police = pickle.load(f)

        # import routes generated on original graph
        with open(f'../data/routes/orig/results_routes_sp_orig_{city}.pkl', 'rb') as f:
            route_fugitive = pickle.load(f)

        routes_labeled = []
        for r, route in enumerate(route_fugitive):
            route_nodes = list(route.values())
            if route_nodes[-1] in escape_nodes:
                routes_labeled.append(route)

        route_data = route_convert(routes_labeled)

        # construct network from fug routes and shortest paths to nodes visited by fug routes
        start_time = time.time()
        G = construct_network(G_orig, routes_labeled, start_police)
        ox.save_graphml(G,f"../data/networks/onthefly_network_{city}.graph.graphml")

        print(time.time()-start_time, 'seconds for graph construction')
        print(len(G_orig.nodes))
        print(len(G.nodes))
        labels = {}
        labels_inv = {}
        for i, node in enumerate(G.nodes()):
            labels[node] = i
            labels_inv[i] = node

        # sort indices on distance to start_fugitive
        if SSR:
            labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted = sort_and_filter_nodes(
                G,
                fugitive_start,
                routes_labeled,
                start_police,
                t_max)
        else:
            labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted = sort_nodes(
                G,
                fugitive_start,
                start_police)

        upper_bounds = []
        for u in range(len(start_police)):
            if len(labels_perunit_sorted[u]) <= 1:
                upper_bounds.append(0.999)
            else:
                upper_bounds.append(len(labels_perunit_sorted[u]) - 0.001)  # different for each unit

        x = 1
        for y in upper_bounds:
            x = x * round(y)
        print('num permutations: ', x)

        num_units = len(start_police)

        # results, intercepted_routes, results_positions = optimize(G, start_police, upper_bounds, num_units, route_data, t_max,
        #          labels, labels_perunit_inv_sorted, SSR)
        #
        # print(city)
        # print(results)
        # print(intercepted_routes)
        #
        # if SSR:
        #     with open(f'./results/with_SSR/onthefly/results_optimization_{city}.pkl', 'wb') as f:
        #         pickle.dump(results, f)
        #
        #     with open(f'./results/with_SSR/onthefly/results_intercepted_routes_{city}.pkl', 'wb') as f:
        #         pickle.dump(intercepted_routes, f)
        #
        #     with open(f'./results/with_SSR/onthefly/results_positions_{city}.pkl', 'wb') as f:
        #         pickle.dump(results_positions, f)
        # else:
        #     with open(f'./results/with_SSR/onthefly/results_optimization_{city}.pkl', 'wb') as f:
        #         pickle.dump(results, f)
        #
        #     with open(f'./results/with_SSR/onthefly/results_intercepted_routes_{city}.pkl', 'wb') as f:
        #         pickle.dump(intercepted_routes, f)
        #
        #     with open(f'./results/with_SSR/onthefly/results_positions_{city}.pkl', 'wb') as f:
        #         pickle.dump(results_positions, f)