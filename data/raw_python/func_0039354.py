def quadrant(xcoord, ycoord):
    """
    Find the quadrant a pair of coordinates are located in

    :type xcoord: integer
    :param xcoord: The x coordinate to find the quadrant for

    :type ycoord: integer
    :param ycoord: The y coordinate to find the quadrant for
    """

    xneg = bool(xcoord < 0)
    yneg = bool(ycoord < 0)
    if xneg is True:
        if yneg is False:
            return 2
        return 3
    if yneg is False:
        return 1
    return 4