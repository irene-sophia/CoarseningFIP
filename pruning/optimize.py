import pickle
import numpy as np
import osmnx as ox
import networkx as nx

from ema_workbench import MultiprocessingEvaluator, SequentialEvaluator
from ema_workbench import RealParameter, ScalarOutcome, Constant, Model
from ema_workbench.em_framework.optimization import ArchiveLogger, SingleObjectiveBorgWithArchive

from panchamy.optimization.sort_and_filter import sort_and_filter_pol_fug_city_pruning as sort_and_filter_nodes
from panchamy.cut_graph_to_boundaries import cut_graph

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
             labels, labels_perunit_inv_sorted):
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
    with MultiprocessingEvaluator(model, n_processes=10) as evaluator:
    # with SequentialEvaluator(model) as evaluator:
        for _ in range(20):  # TODO: change to 5
            convergence_metrics = [
                ArchiveLogger(
                    f"./results/",
                    [l.name for l in model.levers],
                    [o.name for o in model.outcomes if o.kind != o.INFO],
                    base_filename=f"archives.tar.gz"
                ),
            ]

            result = evaluator.optimize(
                algorithm=SingleObjectiveBorgWithArchive,
                nfe=20000,  # TODO: change to 20 000
                searchover="levers",
                convergence=convergence_metrics,
                convergence_freq=100
            )

            result = result.iloc[0]
            print(_, result['num_intercepted'])
            if result['num_intercepted'] >= highest_perf:
                results = result
                highest_perf = result['num_intercepted']

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
    graph_type = 'coarsened'
    # for city in ['Winterswijk', 'Manhattan', 'Utrecht', 'Amsterdam', 'Rotterdam']:
    for city in ['Utrecht']:
        # city = 'Winterswijk'
        # graph_type = 'coarsened'
        print(city)
        G = ox.load_graphml(filepath=f"results/pruned_network_{city}.graph.graphml")

        labels = {}
        labels_inv = {}
        for i, node in enumerate(G.nodes()):
            labels[node] = i
            labels_inv[i] = node

        # import escape nodes
        with open(f'../networks/escape_nodes_{city}.pkl', 'rb') as f:
            escape_nodes = pickle.load(f)

        # import start fugitive
        with open(f'../networks/fugitive_start_{city}.pkl', 'rb') as f:
            fugitive_start = pickle.load(f)

        # import start police: find closest nodes to offices again to avoid that nodes do not exist
        if city == 'Winterswijk':
            police_addresses = [{'lat': 51.96730689392284, 'lon': 6.717868176220198},  # Dingstraat 38
                                {'lat': 51.93473914644306, 'lon': 6.808688143805569},  # Duitsland Oost
                                {'lat': 52.03601213823647, 'lon': 6.822783925913308},  # Duitsland Noord
                                ]
            fugitive_start = ox.nearest_nodes(G, 6.7150548, 51.9741482)
        elif city == 'Utrecht':
            police_addresses = [{'lat': 52.09480988724321, 'lon': 5.112300488054929},  # 'Kroonstraat 25, Utrecht',
                                {'lat': 52.106435888123734, 'lon': 5.081021768548001},
                                {'lat': 52.11657425557721, 'lon': 5.106787541127556},
                                {'lat': 52.077098634761406, 'lon': 5.123281701661107},  # 'Briljantlaan 3, Utrecht',
                                {'lat': 52.07221414683985, 'lon': 5.0946141119890545}  # 'Marco Pololaan 6, Utrecht'
                                ]
            fugitive_start = ox.nearest_nodes(G, 5.123583852763527, 52.09520282139232)
        elif city == 'Manhattan':
            police_addresses = [{'lat': 40.75670564515611, 'lon': -73.97082852600276},  # 51st & 3rd
                                {'lat': 40.74287247577036, 'lon': -73.99856976473754},  # 20th & 7th
                                {'lat': 40.72638720293323, 'lon': -73.98796587163713},  # 5th & 2nd
                                {'lat': 40.712326498711, 'lon': -74.00093381696188},  # Madison
                                {'lat': 40.71627379693516, 'lon': -73.98401898908662},  # Broome & Pitt
                                ]
            fugitive_start = ox.nearest_nodes(G, -73.98959647256116, 40.73585784808693)
        elif city == 'Amsterdam':
            police_addresses = [{'lat': 52.444181727658865, 'lon': 4.839227926343865},  # noord
                                {'lat': 52.30236635114219, 'lon': 4.863404367567513},  # amstelveen
                                {'lat': 52.29749485181312, 'lon': 4.974119183183689},  # ZO
                                {'lat': 52.36406225532534, 'lon': 4.838630215024896},  # west
                                {'lat': 52.3574596444559, 'lon': 4.927086601394641},  # watergraafsmeer
                                ]
            fugitive_start = ox.nearest_nodes(G, 4.9196653961917365, 52.38554053106049)
        elif city == 'Rotterdam':
            police_addresses = [{'lat': 51.94242030923665, 'lon': 4.496820222954058},
                                {'lat': 51.91263550263607, 'lon': 4.4340630196908535},
                                {'lat': 51.90196233470228, 'lon': 4.495848688947844},
                                {'lat': 51.873202510428364, 'lon': 4.482871571074227},
                                {'lat': 51.91661711602705, 'lon': 4.4728571368682175},
                                ]
            fugitive_start = ox.nearest_nodes(G, 4.477156214899593, 51.924154371892584)

        start_police = []
        for pol in police_addresses:
            node = ox.nearest_nodes(G, pol['lon'], pol['lat'])
            start_police.append(node)

        # import delays police
        with open(f'../networks/delays_police_{city}.pkl', 'rb') as f:
            delays_police = pickle.load(f)

        # import routes generated on coarsened graph
        with open(f'../panchamy/simulation/results/results_routes_sp_orig_{city}.pkl', 'rb') as f:
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

        results, intercepted_routes, results_positions = optimize(G, start_police, upper_bounds, num_units, route_data, t_max,
                 labels, labels_perunit_inv_sorted)

        print(results)
        print('DONE: ', city)
        # print(len(intercepted_routes))

        with open(f'./results/pruning_results_optimization_{city}.pkl', 'wb') as f:
            pickle.dump(results, f)

        with open(f'./results/pruning_results_intercepted_routes_{city}.pkl', 'wb') as f:
            pickle.dump(intercepted_routes, f)

        with open(f'./results/pruning_results_positions_{city}.pkl', 'wb') as f:
            pickle.dump(results_positions, f)
