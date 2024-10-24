
def add_meta_info_to_links(links, shape, lookuptable):
    # This function adds meta information to the links table from the shape struct
    # The meta information includes external link id (osm id) and converts the ordinal road type into numerical values based on the lookup table.
    # This assumes the shape struct to contain a numerical column osm_id and a string column type.
    # If column names of your shape file are different, make the necessary changes.

    # Add section details
    all_link_ids = [link['id'] for link in links]
    all_link_ids.extend([max(all_link_ids) * 10] * 5)
    types = [entry['type'] for entry in lookuptable]

    for shape_entry in shape:
        link_ids = shape_entry['link_ids']
        index = [i for i, link_id in enumerate(all_link_ids) if link_id in link_ids]
        type_index = types.index(shape_entry['type'])
        for j in index:
            if 'osm_id' in links[j]:
                links[j]['osm_id'] = list(set(links[j]['osm_id'] + [shape_entry['osm_id']]))
                links[j]['type'] = list(set(links[j]['type'] + [lookuptable[type_index]['id']]))
            else:
                links[j]['osm_id'] = shape_entry['osm_id']
                links[j]['type'] = lookuptable[type_index]['id']

    ids = [i for i, link in enumerate(links) if not link.get('osm_id')]
    links = [link for i, link in enumerate(links) if i not in ids]

    return links