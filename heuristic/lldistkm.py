import math

def lldistkm(latlon1, latlon2):
    # Distance calculation based on Haversine formula
    radius = 6371  # Earth's radius in km
    lat1 = latlon1[0] * math.pi / 180
    lat2 = latlon2[0] * math.pi / 180
    lon1 = latlon1[1] * math.pi / 180
    lon2 = latlon2[1] * math.pi / 180
    deltaLat = lat2 - lat1
    deltaLon = lon2 - lon1
    a = math.sin(deltaLat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(deltaLon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d1km = radius * c  # Haversine distance

    # Distance calculation based on Pythagorean theorem
    x = deltaLon * math.cos((lat1 + lat2) / 2)
    y = deltaLat
    d2km = radius * math.sqrt(x * x + y * y)  # Pythagorean distance

    return d1km, d2km