def flipcoords(xcoord, ycoord, axis):
    """
    Flip the coordinates over a specific axis, to a different quadrant

    :type xcoord: integer
    :param xcoord: The x coordinate to flip

    :type ycoord: integer
    :param ycoord: The y coordinate to flip

    :type axis: string
    :param axis: The axis to flip across. Could be 'x' or 'y'
    """

    axis = axis.lower()
    if axis == 'y':
        if xcoord > 0:
            return str(xcoord - xcoord - xcoord) + ', ' + str(ycoord)
        elif xcoord < 0:
            return str(xcoord + abs(xcoord) * 2) + ', ' + str(ycoord)
        elif xcoord == 0:
            return str(xcoord) + ', ' + str(ycoord)
        raise ValueError(
            "The X coordinate is neither larger, smaller or the same as 0.")

    elif axis == 'x':
        if ycoord > 0:
            return str(xcoord) + ', ' + str(ycoord - ycoord - ycoord)
        elif ycoord < 0:
            return str(ycoord + abs(ycoord) * 2) + ', ' + str(xcoord)
        elif ycoord == 0:
            return str(xcoord) + ', ' + str(ycoord)
        raise ValueError(
            "The Y coordinate is neither larger, smaller or the same as 0.")
    raise ValueError("Invalid axis. Neither x nor y was specified.")