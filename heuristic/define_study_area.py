def define_study_area(vertex, lat_min, lat_max, lon_min, lon_max):
    xv = [lat_min, lat_min, lat_max, lat_max, lat_min]
    yv = [lon_min, lon_max, lon_max, lon_min, lon_min]
    coords = [[v.coordinates[0], v.coordinates[1]] for v in vertex]
    xq = [c[0] for c in coords]
    yq = [c[1] for c in coords]
    in_polygon = [point_in_polygon(c[0], c[1], xv, yv) for c in coords]
    exempt_ids = [v.id for v, is_in in zip(vertex, in_polygon) if is_in]
    return exempt_ids

def point_in_polygon(x, y, xv, yv):
    n = len(xv)
    inside = False
    p1x, p1y = xv[0], yv[0]
    for i in range(n + 1):
        p2x, p2y = xv[i % n], yv[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside