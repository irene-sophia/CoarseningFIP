import logging
import pickle
import random
import time

import networkx as nx
import osmnx as ox

from basic_logger import get_module_logger
from plot_results import plot_routes
from pydsol.core.experiment import SingleReplication
from pydsol.core.simulator import DEVSSimulatorFloat
from model_sp import FugitiveModel
from panchamy.cut_graph_to_boundaries import cut_graph

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s (%(name)s - %(filename)s: line %(lineno)s)')

logger = get_module_logger(__name__, level=logging.INFO)

def manhattan_graph(N):
    G = nx.grid_2d_graph(N, N)

    # set x and y
    i = 0
    locations = {}
    for u, v in G.nodes():
        locations[(u, v)] = {"x": u, "y": v}
        i += 1
    nx.set_node_attributes(G, locations)

    # set travel time (unidistant)
    travel_time = 1
    nx.set_edge_attributes(G, travel_time, "travel_time")

    pos = dict((n, n) for n in G.nodes())
    labels = dict(((i, j), i * N + j) for i, j in G.nodes())  #
    labels_inv = dict((i * N + j, (i, j)) for i, j in G.nodes())

    return G, labels, labels_inv, pos



if __name__ == '__main__':
    mode = 'shortest_path'
    seed = 112
    random.seed(seed)
    noise = 0.02

    graph_type = 'coarsened'  # orig / coarsened
    run_length = 1800

    # for weight_type in ['type', 'betweenness']:
    for weight_type in ['betweenness']:
        # for city in ['Rotterdam']:
        for city in ['Manhattan', 'Amsterdam', 'Rotterdam', 'Winterswijk', 'Utrecht']:
        # for city in ['Amsterdam']:
        # for city in ['Utrecht', 'Manhattan', 'Winterswijk']:
            for pruning in [0, 1]:
                for iterations in [1, 1000]:
                    for threshold in [0, 1000]:
        #     for pruning in [1]:
        #         for iterations in [1000]:
        #             for threshold in [1000]:

                        with open(f'../../data/networks/escape_nodes_{city}.pkl', 'rb') as f:
                            escape_nodes = pickle.load(f)

                        with open(f'../../data/networks/fugitive_start_{city}.pkl', 'rb') as f:
                            fugitive_start = pickle.load(f)

                        if graph_type == 'coarsened':
                            graph = ox.load_graphml(
                                filepath=f"../../data/networks/panchamy_{weight_type}_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.graph.graphml")

                        elif graph_type == 'orig':
                            graph = ox.load_graphml(filepath=f"../networks/{city}.graph.graphml")
                        print(len(graph.nodes))
                        if city != 'Amsterdam':
                            graph = cut_graph(graph, city)
                        print(len(graph.nodes))


                        route_fugitive = []
                        u = graph.to_undirected()
                        while len(route_fugitive) < 1000:
                            for escape_node in escape_nodes:
                                if escape_node in graph.nodes():
                                    try:
                                        path = nx.shortest_path(graph, fugitive_start, escape_node, 'travel_time')
                                        # [escape_routes.append(path) for path in nx.all_simple_paths(G, fugitive_start, escape_node)]
                                        route_fugitive.append(path)
                                    except:

                                        if nx.has_path(u, fugitive_start, escape_node):
                                            path = nx.shortest_path(u, fugitive_start, escape_node, 'travel_time')
                                            route_fugitive.append(path)
                                        continue
                        route_fugitive = route_fugitive[:1000]

                        simulator = DEVSSimulatorFloat("sim")
                        model = FugitiveModel(simulator=simulator,
                                              input_params={'seed': seed,
                                                            'graph': graph,
                                                            'start_fugitive': fugitive_start,
                                                            'route_fugitive': route_fugitive,
                                                            'num_fugitive_routes': len(route_fugitive),
                                                            'jitter': noise,
                                                            'escape_nodes': escape_nodes,
                                                            },
                                              seed=seed)
                        print('starting simulation')

                        replication = SingleReplication(str(0), 0.0, 0.0, run_length)
                        # experiment = Experiment("test", simulator, sim_model, 0.0, 0.0, 700, nr_replications=5)
                        simulator.initialize(model, replication)
                        simulator.start()
                        # Python wacht niet todat de simulatie voorbij is, vandaar deze while loop
                        while simulator.simulator_time < run_length:
                            time.sleep(0.01)

                        routes = model.get_output_statistics()

                        if graph_type == 'coarsened':
                            with open(f'results/results_routes_sp_panchamy_{weight_type}_{city}_pruning{pruning}_iter{iterations}_threshold{threshold}.pkl', 'wb') as f:
                                pickle.dump(routes, f)
                        elif graph_type == 'orig':
                            with open(f'results/results_routes_sp_{graph_type}_{city}.pkl', 'wb') as f:
                                pickle.dump(routes, f)

                        # plot_routes(city, weight_type, pruning, iterations, threshold)

                        model.reset_model()
