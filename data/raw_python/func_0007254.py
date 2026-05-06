def osm_filter(network_type):
    """
    Create a filter to query Overpass API for the specified OSM network type.

    Parameters
    ----------
    network_type : string, {'walk', 'drive'} denoting the type of street
    network to extract

    Returns
    -------
    osm_filter : string
    """
    filters = {}

    # drive: select only roads that are drivable by normal 2 wheel drive
    # passenger vehicles both private and public
    # roads. Filter out un-drivable roads and service roads tagged as parking,
    # driveway, or emergency-access
    filters['drive'] = ('["highway"!~"cycleway|footway|path|pedestrian|steps'
                        '|track|proposed|construction|bridleway|abandoned'
                        '|platform|raceway|service"]'
                        '["motor_vehicle"!~"no"]["motorcar"!~"no"]'
                        '["service"!~"parking|parking_aisle|driveway'
                        '|emergency_access"]')

    # walk: select only roads and pathways that allow pedestrian access both
    # private and public pathways and roads.
    # Filter out limited access roadways and allow service roads
    filters['walk'] = ('["highway"!~"motor|proposed|construction|abandoned'
                       '|platform|raceway"]["foot"!~"no"]'
                       '["pedestrians"!~"no"]')

    if network_type in filters:
        osm_filter = filters[network_type]
    else:
        raise ValueError('unknown network_type "{}"'.format(network_type))

    return osm_filter