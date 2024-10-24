import osmnx as ox
import geopandas as gpd
import shapely
import networkx as nx

def cut_graph(G, city):
    nodes_gdf, streets = ox.graph_to_gdfs(G, nodes=True, edges=True,
                                          node_geometry=False, fill_edge_geometry=True)
    streets = streets.to_crs(4326)

    if city == 'Utrecht':
        # import geopackage file in geopandas
        outline = gpd.read_file("../data/networks/Utrecht.geojson", layer=0)
        # nld = gpd.read_file("graphs/FLEE/Netherlands_shapefile/nl_1km.shp")
        # outline = outline.to_crs(4326)

        # multiLines = shapely.geometry.MultiLineString([x.exterior for x in nld.geometry])

        # lines = streets.geometry.unary_union
        # # intersection = lines.intersection(nld.geometry[0])
        # intersectionn = gpd.sjoin(streets, outline)

    if city == 'Rotterdam':
        # import geopackage file in geopandas
        outline = gpd.read_file("../data/networks/rotterdam1.geojson", layer=0)

    if city == 'Manhattan':
        outline = gpd.read_file("../data/networks/manhattan-island_1372.geojson")
        # nld = gpd.read_file("graphs/FLEE/Netherlands_shapefile/nl_1km.shp")
        # outline = outline.to_crs(4326)

        # # multiLines = mh.geometry[0].exterior
        # lines = streets.geometry.unary_union  # same
        # # intersection = lines.intersection(mh.geometry[0])  # same
        # intersectionn = gpd.sjoin(streets, outline)  # same


    if city == 'Winterswijk':
        outline = gpd.read_file("../data/networks/BestuurlijkeGebieden_2023.gpkg", layer=1)
        # nld = gpd.read_file("graphs/FLEE/Netherlands_shapefile/nl_1km.shp")

    outline = outline.to_crs(4326)
    # multiLines = shapely.geometry.MultiLineString([x.exterior for x in nld.geometry[0].geoms])
    lines = streets.geometry.unary_union
    # intersection = lines.intersection(nld.geometry[0])
    for col in list(streets.columns):
        if '_left' in col:
            del streets[col]
        elif '_right' in col:
            del streets[col]
    # if 'index_right' in list(streets.columns):
    #     del streets['index_right']
    # if 'identificatie_left' in list(streets.columns):
    #     del streets['identificatie_left']
    # if 'identificatie_right' in list(streets.columns):
    #     del streets['identificatie_right']
    # if 'name_left' in list(streets.columns):
    #     del streets['name_left']
    # if 'naam_left' in list(streets.columns):
    #     del streets['naam_left']
    # if 'code_left' in list(streets.columns):
    #     del streets['code_left']

    intersectionn = gpd.sjoin(streets, outline)

    unique_osmids_u = list(intersectionn.index.unique(level=0))
    unique_osmids_v = list(intersectionn.index.unique(level=1))
    unique_osmids = list(set(unique_osmids_u) | set(unique_osmids_v))
    # nodes_gdf.iloc[unique_osmids]
    nodes_gdf_intersection = nodes_gdf[nodes_gdf.index.isin(unique_osmids)]

    G_intersection = ox.graph_from_gdfs(nodes_gdf_intersection, intersectionn)

    N_subs = 1  # Number of biggest islands you want to keep
    G_sub = []
    largest_components = []
    for i in range(N_subs):
        largest_components.append(sorted(nx.weakly_connected_components(G_intersection), key=len, reverse=True)[i])
        G_sub.append(G_intersection.subgraph(largest_components[i]))

    G_cc = nx.compose_all(G_sub)

    return G_cc
