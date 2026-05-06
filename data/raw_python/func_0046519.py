def eere_station(station_code):
    """Station information.

    Args:
        station_code (str): station code.

    Returns (dict): station information
    """
    with open(env.SRC_PATH + '/eere_meta.csv') as eere_meta:
        stations = csv.DictReader(eere_meta)
        for station in stations:
            if station['station_code'] == station_code:
                return station
    raise KeyError('station not found')