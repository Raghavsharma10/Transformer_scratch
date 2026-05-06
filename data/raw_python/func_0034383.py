def extract_stations(page):
    '''Extract bus stations from routine page.

    :param page: crawled page.
    '''
    stations = [_(station.value) for station in page('.stateName')]
    return {
        'terminal': {
            stations[0]: list(reversed(stations)),
            stations[-1]: stations
        },
        'stations': stations
    }