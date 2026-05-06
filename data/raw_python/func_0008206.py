def _filter_invalid_routes(routes, board, railroad):
    """
    Given a collection of routes, returns a new set containing only valid routes. Invalid routes removed:
    - contain less than 2 cities, or
    - go through Chicago using an impassable exit
    - only contain Chicago as a station, but don't use the correct exit path

    This fltering after the fact keeps the path finding algorithm simpler. It allows groups of 3 cells to be considered
    (important for the Chicago checks), which would be tricky, since the algorithm operates on pairs of cells (at the
    time of writing).
    """
    chicago_space = board.get_space(CHICAGO_CELL)

    chicago_neighbor_cells = [cell for cell in CHICAGO_CELL.neighbors.values() if cell != CHICAGO_CONNECTIONS_CELL]
    stations = board.stations(railroad.name)

    # A sieve style filter. If a condition isn't met, iteration continues to the next item. Items meeting all conditions
    # are added to valid_routes at the end of the loop iteration.
    valid_routes = set()
    for route in routes:
        # A route must connect at least 2 cities.
        if len(route.cities) < 2:
            continue

        # A route cannot run from east to east
        if isinstance(route.cities[0], EastTerminalCity) and isinstance(route.cities[-1], EastTerminalCity):
            continue

        # If the route goes through Chicago and isn't [C5, D6], ensure the path it took either contains its station or is unblocked
        if route.contains_cell(CHICAGO_CONNECTIONS_CELL) and len(route.cities) != 2:
            # Finds the subroute which starts at Chicago and is 3 tiles long. That is, it will go [C5, D6, chicago exit]
            all_chicago_subroutes = [subroute for subroute in route.subroutes(CHICAGO_CONNECTIONS_CELL) if len(subroute) == 3]
            chicago_subroute = all_chicago_subroutes[0] if all_chicago_subroutes else None
            for cell in chicago_neighbor_cells:
                chicago_exit = chicago_subroute and chicago_subroute.contains_cell(cell)
                if chicago_exit and chicago_space.passable(cell, railroad):
                    break
            else:
                continue

        # Each route must contain at least 1 station
        stations_on_route = [station for station in stations if route.contains_cell(station.cell)]
        if not stations_on_route:
            continue
        # If the only station is Chicago, the path must be [D6, C5], or exit through the appropriate side.
        elif [CHICAGO_CELL] == [station.cell for station in stations_on_route]:
            exit_cell = board.get_space(CHICAGO_CELL).get_station_exit_cell(stations_on_route[0])
            chicago_exit_route = Route.create([chicago_space, board.get_space(exit_cell)])
            if not (len(route) == 2 and route.contains_cell(CHICAGO_CONNECTIONS_CELL)) and not route.overlap(chicago_exit_route):
                continue

        valid_routes.add(route)

    return valid_routes