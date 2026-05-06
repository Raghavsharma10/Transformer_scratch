def query_realtime_routine(bus_name, cur_station=None):
    '''Get real time routine.

    TODO support fuzzy matching.

    :param bus_name: the routine name of the bus.
    :param cur_station: current station, deaults to starting station
                        of the routine.
    '''
    routines = query_routines(bus_name)
    if not routines:
        return

    rv = []
    for routine in routines:
        bid = routine['bid']
        _cur_station = cur_station or routine['starting_station']
        page = _get_realtime_page(bus_name, bid, _cur_station)
        rv.append(extract_bus_routine(page))

    return rv