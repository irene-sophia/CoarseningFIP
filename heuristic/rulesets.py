import numpy as np
from collapse_node import collapse_node


def rulesets(links, nodes, weights, i, max_link_id, params, pred, succ):
    var_threshold = params['threshold']
    constraint_links = params['constraint_links']
    pruning = params['pruning']
    flag_intersection = params['flag_intersection']

    incoming = pred
    outgoing = succ
    node_id = str(i)

    if flag_intersection == 0:
        shortcut_size = len(incoming) + len(outgoing)
    else:
        shortcut_size = 2

    if not incoming or not outgoing:
        if pruning == 1:
            if not incoming:
                outgoing_index = [link_data['id'] for link, link_data in links.items() if link in outgoing]
                # outgoing_index = [index for index, link in enumerate(links) if link['id'] in outgoing]
                # weights = [weights[index] for index in outgoing_index]
                links = {k:v for k,v in links.items() if v['id'] not in outgoing_index}
                #links = [link for index, link in enumerate(links) if link['id'] in outgoing]
            else:
                incoming_index = [link_data['id'] for link, link_data in links.items() if link in incoming]
                # weights = [weights[index] for index in incoming_index]
                links = {k:v for k,v in links.items() if v['id'] not in incoming_index}
        i += 1
    else:
        pairs = [(p, q) for p in incoming for q in outgoing]
        if constraint_links == 1 and len(pairs) >= shortcut_size:
            i += 1
        else:
            incoming_index = [link_data['id'] for link, link_data in links.items() if link in incoming]
            outgoing_index = [link_data['id'] for link, link_data in links.items() if link in outgoing]
            incoming_weight = [weights[index] for index in incoming_index]
            outgoing_weight = [weights[index] for index in outgoing_index]
            weight_pairs = [[p, q] for p in incoming_weight for q in outgoing_weight]  # this is basically 'combinations'
            if any(np.var(pair) > var_threshold for pair in weight_pairs):  # TODO: does this make sense?
                i += 1
            else:
                links, nodes, weights, max_link_id = collapse_node(links, nodes, weights, node_id, max_link_id, incoming_index, outgoing_index, pruning)
                i += 1

    return links, nodes, weights, max_link_id, i
