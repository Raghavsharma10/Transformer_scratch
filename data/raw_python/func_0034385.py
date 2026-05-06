def extract_bus_routine(page):
    '''Extract bus routine information from page.

    :param page: crawled page.
    '''
    if not isinstance(page, pq):
        page = pq(page)

    stations = extract_stations(page)
    return {
        # Routine name.
        'name': extract_routine_name(page),

        # Bus stations.
        'stations': stations,

        # Current routine.
        'current': extract_current_routine(page, stations)
    }