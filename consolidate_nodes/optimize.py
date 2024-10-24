import pickle
import numpy as np
import osmnx as ox
import networkx as nx
import ast

from ema_workbench import MultiprocessingEvaluator, SequentialEvaluator
from ema_workbench import RealParameter, ScalarOutcome, Constant, Model
from ema_workbench.em_framework.optimization import ArchiveLogger, SingleObjectiveBorgWithArchive

from panchamy.optimization.sort_and_filter import sort_and_filter_pol_fug_city_pruning as sort_and_filter_nodes
from panchamy.cut_graph_to_boundaries import cut_graph

import logging
from functools import wraps
import time

# logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# logger_timer = logging.getLogger(__name__)
# logging.basicConfig(filename='example.txt', encoding='utf-8', level=logging.INFO)

# def timeit(func):
#     @wraps(func)
#     def timeit_wrapper(*args, **kwargs):
#         start_time = time.perf_counter()
#         result = func(*args, **kwargs)
#         end_time = time.perf_counter()
#         total_time = end_time - start_time
#         #print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
#         # print(f'FIP Function took {total_time:.4f} seconds')
#         return result
#     return timeit_wrapper


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
        travel_time_to_target = nx.shortest_path_length(graph,
                                                      source=police_start[u],
                                                      target=associated_node,
                                                      weight='travel_time',
                                                      method='bellman-ford')
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


def optimize(graph, city, tolerance, police_start, upper_bounds, num_units, route_data, t_max,
             labels, labels_perunit_inv_sorted):
    model = Model(f"FIPEMA", function=FIP_func)

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
    with MultiprocessingEvaluator(model, n_processes=10) as evaluator:
    # with SequentialEvaluator(model) as evaluator:
        for seed in range(1):  # TODO: change to 10
            convergence_metrics = [
                ArchiveLogger(
                    f".",
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes if o.kind != o.INFO],
                    base_filename=f"archives.tar.gz"
                ),
            ]

            result = evaluator.optimize(
                algorithm=SingleObjectiveBorgWithArchive,
                nfe=2000,  # TODO: change to 100 000
                searchover="levers",
                convergence=convergence_metrics,
                convergence_freq=100,
                LoggingName=f'{city}: {tolerance}'
            )

            result = result.iloc[0]
            # print(seed, result['num_intercepted'])
            if result['num_intercepted'] >= highest_perf:
                results = result
                highest_perf = result['num_intercepted']

            results_positions_ = []
            results_positions_labeled_ = []
            for u, start in enumerate(police_start):
                results_positions_labeled_.append(result[f'pi_{u}'])
                results_positions_.append(labels_perunit_inv_sorted[u][int(np.floor(result[f'pi_{u}']))])
            # print(results_positions)

            routes_intercepted_ = get_intercepted_routes(graph, route_data, results_positions_, police_start)

            G_orig = ox.load_graphml(filepath=f"../data/networks/{city}.graph.graphml")
            with open(f'../data/routes/orig/results_routes_sp_orig_{city}.pkl', 'rb') as f:
                route_fugitive = pickle.load(f)
            routes_labeled = []
            for r, route in enumerate(route_fugitive):
                route_nodes = list(route.values())
                if route_nodes[-1] in escape_nodes:
                    routes_labeled.append(route)
            route_data = route_convert(routes_labeled)
            with open(f'../data/networks/start_police_{city}.pkl', 'rb') as f:
                start_police = pickle.load(f)
            cross_eval = get_intercepted_routes(G_orig, route_data, results_positions_, start_police)
            print('CROSS EVAL: ', len(cross_eval))


            with open(f'./results/optimization/consolidated_results_optimization_{city}_{tolerance}_seed{seed}.pkl', 'wb') as f:
                pickle.dump(result, f)

            with open(f'./results/optimization/consolidated_results_intercepted_routes_{city}_{tolerance}_seed{seed}.pkl', 'wb') as f:
                pickle.dump(routes_intercepted_, f)

            with open(f'./results/optimization/consolidated_results_positions_{city}_{tolerance}_seed{seed}.pkl', 'wb') as f:
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

    # for city in ['Winterswijk', 'Manhattan', 'Utrecht', 'Amsterdam', 'Rotterdam']:
    for city in ['Winterswijk', 'Manhattan']:
        # for tolerance in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for tolerance in [40]:
            # logger_timer.info(f"{city}: tolerance: {tolerance}")

            filepath = f"../data/networks/consolidated_network_{city}_{tolerance}.graph.graphml"
            G = ox.load_graphml(filepath=filepath)

            labels = {}
            labels_inv = {}
            for i, node in enumerate(G.nodes()):
                labels[node] = i
                labels_inv[i] = node

            # import escape nodes
            with open(f'../data/networks/escape_nodes_{city}.pkl', 'rb') as f:
                escape_nodes = pickle.load(f)

            with open(f'../data/networks/fugitive_start_{city}.pkl', 'rb') as f:
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

            with open(f'../data/networks/start_police_{city}.pkl', 'rb') as f:
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

            # import delays police
            with open(f'../data/networks/delays_police_{city}.pkl', 'rb') as f:
                delays_police = pickle.load(f)

            # import routes generated on coarsened graph
            # with open(f'./results/routes/results_routes_sp_consolidated_{city}_{tolerance}.pkl', 'rb') as f:
            #     route_fugitive = pickle.load(f)

            with open(f'../data/routes/orig/results_routes_sp_orig_{city}.pkl', 'rb') as f:
                route_fugitive = pickle.load(f)

            routes_labeled = []
            for r, route in enumerate(route_fugitive):
                route_nodes = list(route.values())
                if route_nodes[-1] in escape_nodes:
                    routes_labeled.append(route)

            route_data = route_convert(routes_labeled)

            # sort indices on distance to start_fugitive
            labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted = sort_and_filter_nodes(
                G,
                fugitive_start,
                routes_labeled,
                start_police,
                t_max)

            # with open(f'../results/labels_perunit_sorted_{city}.pkl', 'wb') as f:
            #     pickle.dump(labels_perunit_sorted, f)
            # with open(f'../results/labels_perunit_inv_sorted_{city}.pkl', 'wb') as f:
            #     pickle.dump(labels_perunit_inv_sorted, f)
            # with open(f'../results/labels_full_sorted_{city}.pkl', 'wb') as f:
            #     pickle.dump(labels_full_sorted, f)

            # with open(f'../results/labels_perunit_sorted_{city}.pkl', 'rb') as f:
            #     labels_perunit_sorted = pickle.load(f)
            # with open(f'../results/labels_perunit_inv_sorted_{city}.pkl', 'rb') as f:
            #     labels_perunit_inv_sorted = pickle.load(f)
            # with open(f'../results/labels_full_sorted_{city}.pkl', 'rb') as f:
            #     labels_full_sorted = pickle.load(f)

            upper_bounds = []
            for u in range(len(start_police)):
                if len(labels_perunit_sorted[u]) <= 1:
                    upper_bounds.append(0.999)
                else:
                    upper_bounds.append(len(labels_perunit_sorted[u]) - 0.001)  # different for each unit

            num_units = len(start_police)

            results, intercepted_routes, results_positions = optimize(G, city, tolerance, start_police, upper_bounds, num_units, route_data, t_max,
                     labels, labels_perunit_inv_sorted)

            print(results)

            G_orig = ox.load_graphml(filepath=f"../data/networks/{city}.graph.graphml")
            with open(f'../data/routes/orig/results_routes_sp_orig_{city}.pkl', 'rb') as f:
                route_fugitive = pickle.load(f)
            routes_labeled = []
            for r, route in enumerate(route_fugitive):
                route_nodes = list(route.values())
                if route_nodes[-1] in escape_nodes:
                    routes_labeled.append(route)
            route_data = route_convert(routes_labeled)
            with open(f'../data/networks/start_police_{city}.pkl', 'rb') as f:
                start_police = pickle.load(f)
            cross_eval = get_intercepted_routes(G_orig, route_data, results_positions, start_police)
            print('CROSS EVAL: ', len(cross_eval))

            print('DONE: ', city, tolerance)
            # print(len(intercepted_routes))

            with open(f'./results/optimization/consolidated_results_optimization_{city}_{tolerance}.pkl', 'wb') as f:
                pickle.dump(results, f)

            with open(f'./results/optimization/consolidated_results_intercepted_routes_{city}_{tolerance}.pkl', 'wb') as f:
                pickle.dump(intercepted_routes, f)

            with open(f'./results/optimization/consolidated_results_positions_{city}_{tolerance}.pkl', 'wb') as f:
                pickle.dump(results_positions, f)
