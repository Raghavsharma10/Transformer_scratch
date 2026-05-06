def extract_current_routine(page, stations):
    '''Extract current routine information from page.

    :param page: crawled page.
    :param stations: bus stations list. See `~extract_stations`.
    '''
    current_routines = CURRENT_ROUTINE_PATTERN.findall(page.text())
    if not current_routines:
        return

    terminal_station = stations['stations'][-1]
    for routine in current_routines:
        if _(routine[0]) == terminal_station:
            distance = int(routine[1])
    stations_to_this_dir = stations['terminal'][terminal_station]

    waiting_station = _(page('.now .stateName').val())
    idx = stations_to_this_dir.index(waiting_station)
    bus_station = stations_to_this_dir[idx - distance + 1]

    return {
        'destinate_station': terminal_station,
        'bus_station': bus_station,
        'waiting_station': waiting_station,
        'distance': distance
    }