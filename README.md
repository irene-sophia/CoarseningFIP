This repository accompanies the paper "Graph Coarsening for Fugitive Interception".

- Directories _pruning_, _consolidate\_nodes_, _heuristic_, and _onthefly_ contain the algorithms for graph coarsening. The _heuristic_ coarsening algorithm is the Python implementation of the algorithm proposed by Krishnakumari et al. (2020) and can also be found at https://github.com/irene-sophia/HeuristicCoarsening. 
- The directory _HPC\_results_ contains the results of the experiments for all five road networks for all algorithms.
- The directory _analysis_ contains the notebooks that analyze the experiments, also including the cross-evaluation, counting the numbers of nodes in each network, and the timing experiments. Most plots in the paper are generated in the _compare_methods.ipynb_ notebook.
- The directory _data_ contains the coarsened networks resulting from each algorithm, and the simulated routes.
- The directory _route\_simulation_ contains the route generation code.
