from ismember_custom import ismember_custom


def create_vertex_table(shape):
    # This function creates vertex or node table from the shape struct.
    # This assumes the shape struct to contain column X and Y with coordinates.
    # If column names of your shape file is different, change where there is a <-
    # also add the node id to the shape struct

    # create vertex or nodes from shape file
    all_vertex = []
    for i in range(len(shape)):
        all_vertex.append([shape[i]['X'][0], shape[i]['Y'][0]])  # <- X and Y coordinates
        all_vertex.append([shape[i]['X'][-2], shape[i]['Y'][-2]])  # <- X and Y coordinates

    unique_vertex = list(set(map(tuple, all_vertex)))
    unique_vertex = [list(vertex) for vertex in unique_vertex if not all(math.isnan(coord) for coord in vertex)]

    vertex = []
    for i in range(len(unique_vertex)):
        vertex.append({'id': i, 'coordinates': unique_vertex[i]})

    # add vertex id to shape
    all_vertex = [vertex['coordinates'] for vertex in vertex]
    all_vertex_ids = [vertex['id'] for vertex in vertex]
    all_vertex_sorted = sorted(all_vertex)
    sortInds = sorted(range(len(all_vertex)), key=lambda k: all_vertex[k])
    all_vertex_ids = [all_vertex_ids[i] for i in sortInds]

    # The last elements are not searched
    all_vertex_sorted.append([(max(coord for vertex in all_vertex for coord in vertex) * 10)] * 2)

    for i in range(len(shape)):
        current = []
        current.append([shape[i]['X'][0], shape[i]['Y'][0]])  # <- X and Y coordinates
        current.append([shape[i]['X'][-2], shape[i]['Y'][-2]])  # <- X and Y coordinates
        current = [coord for coord in current if not all(math.isnan(c) for c in coord)]
        index = ismember_custom(current, all_vertex_sorted)
        shape[i]['vertex_ids'] = [all_vertex_ids[i] for i in index]

    return shape
