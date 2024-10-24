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
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import pandas as pd

if __name__ == '__main__':
    t_max = 1800

    results_permutations = []
    for city in ['Winterswijk', 'Manhattan', 'Utrecht', 'Amsterdam', 'Rotterdam']:
    # for city in ['Rotterdam']:
        for tolerance in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    #     for tolerance in [0]:
        # for tolerance in [40, 15, 35, 20, 30, 25]:
            filepath = f"results/networks/consolidated_network_{city}_{tolerance}.graph.graphml"
            G = ox.load_graphml(filepath=filepath)

            # filepath = f"../networks/{city}.graph.graphml"
            # G = ox.load_graphml(filepath=filepath)

            labels = {}
            labels_inv = {}
            for i, node in enumerate(G.nodes()):
                labels[node] = i
                labels_inv[i] = node

            # import escape nodes
            with open(f'../networks/escape_nodes_{city}.pkl', 'rb') as f:
                escape_nodes = pickle.load(f)

            with open(f'../networks/fugitive_start_{city}.pkl', 'rb') as f:
                fugitive_start = pickle.load(f)
            if fugitive_start not in G.nodes:
                for node, data in G.nodes(data=True):
                    if isinstance(data['osmid_original'], int):
                        if fugitive_start == data['osmid_original']:
                            # print(fugitive_start)
                            fugitive_start = node
                            # print(fugitive_start)
                            break
                    elif isinstance(data['osmid_original'], str):
                        if isinstance(ast.literal_eval(data['osmid_original']), int):
                            if fugitive_start == data['osmid_original']:
                                # print(fugitive_start)
                                fugitive_start = node
                                # print(fugitive_start)
                                break
                        elif isinstance(data['osmid_original'], list):
                            if fugitive_start in ast.literal_eval(data['osmid_original']):
                                # print(fugitive_start)
                                fugitive_start = node
                                # print(fugitive_start)
                                break
                    elif isinstance(data['osmid_original'], list):
                        if fugitive_start in data['osmid_original']:
                            # print(fugitive_start)
                            fugitive_start = node
                            # print(fugitive_start)
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
                                # print(i, pol, node)
                        elif isinstance(data['osmid_original'], list):
                            if pol in data['osmid_original']:
                                start_police[i] = node
                                # print(i, pol, node)
                        elif isinstance(data['osmid_original'], str):
                            if isinstance(ast.literal_eval(data['osmid_original']), int):
                                if pol == ast.literal_eval(data['osmid_original']):
                                    start_police[i] = node
                                    # print(i, pol, node)
                            elif isinstance(ast.literal_eval(data['osmid_original']), list):
                                if fugitive_start in ast.literal_eval(data['osmid_original']):
                                    start_police[i] = node
                                    # print(i, pol, node)

            # import delays police
            with open(f'../networks/delays_police_{city}.pkl', 'rb') as f:
                delays_police = pickle.load(f)

            # import routes generated on coarsened graph
            with open(f'./results/routes/results_routes_sp_consolidated_{city}_{tolerance}.pkl', 'rb') as f:
                route_fugitive = pickle.load(f)
            # with open(f'../panchamy/simulation/results/results_routes_sp_orig_{city}.pkl', 'rb') as f:
            #     route_fugitive = pickle.load(f)

            routes_labeled = []
            for r, route in enumerate(route_fugitive):
                route_nodes = list(route.values())
                if route_nodes[-1] in escape_nodes:
                    routes_labeled.append(route)

            # sort indices on distance to start_fugitive
            labels_perunit_sorted, labels_perunit_inv_sorted, labels_full_sorted = sort_and_filter_nodes(
                G,
                fugitive_start,
                routes_labeled,
                start_police,
                t_max)

            upper_bounds = []
            for u in range(len(start_police)):
                if len(labels_perunit_sorted[u]) <= 1:
                    upper_bounds.append(0.999)
                else:
                    upper_bounds.append(len(labels_perunit_sorted[u]) - 0.001)  # different for each unit

            x = 1
            for y in upper_bounds:
                x = x * round(y)
            print('num permutations: ', city, tolerance, x)
            results_permutations.append(x)

    print(results_permutations)
    data = pd.read_excel('./node reduction.xlsx')
    data['permutations coarsened'] = results_permutations
    data['% permutation reduction '] = (data['permutations coarsened'] / data['permutations orig']) * 100
    data.to_excel('./node reduction.xlsx', index=False)


