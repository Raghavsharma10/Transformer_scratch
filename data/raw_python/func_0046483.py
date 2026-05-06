def get_distance_function(distance):
    """
    Returns the distance function from the string name provided

    :param distance: The string name of the distributions
    :return:
    """
    # If we provided distance function ourselves, use it
    if callable(distance):
        return distance
    try:
        return _supported_distances_lookup()[distance]
    except KeyError:
        raise KeyError('Unsupported distance function {0!r}'.format(distance.lower()))