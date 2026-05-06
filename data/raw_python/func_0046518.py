def closest_eere(latitude, longitude):
    """Find closest station from the new(er) list.

    Warning: There may be some errors with smaller non US stations.

    Args:
        latitude (float)
        longitude (float)

    Returns:
        tuple (station_code (str), station_name (str))

    """
    with open(env.SRC_PATH + '/eere_meta.csv') as eere_meta:
        stations = csv.DictReader(eere_meta)
        d = 9999
        station_code = ''
        station_name = ''
        for station in stations:
            new_dist = great_circle((latitude, longitude),
                                    (float(station['latitude']),
                                     float(station['longitude']))).miles
            if new_dist <= d:
                d = new_dist
                station_code = station['station_code']
                station_name = station['weather_station']
        return station_code, station_name
    raise KeyError('station not found')