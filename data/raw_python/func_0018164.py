def _rotate_point(point, angle, ishape, rshape, reverse=False):
    """Transform a point from original image coordinates to rotated image
    coordinates and back. It assumes the rotation point is the center of an
    image.

    This works on a simple rotation transformation::

        newx = (startx) * np.cos(angle) - (starty) * np.sin(angle)
        newy = (startx) * np.sin(angle) + (starty) * np.cos(angle)

    It takes into account the differences in image size.

    Parameters
    ----------
    point : tuple
        Point to be rotated, in the format of ``(x, y)`` measured from
        origin.

    angle : float
        The angle in degrees to rotate the point by as measured
        counter-clockwise from the X axis.

    ishape : tuple
        The shape of the original image, taken from ``image.shape``.

    rshape : tuple
        The shape of the rotated image, in the form of ``rotate.shape``.

    reverse : bool, optional
        Transform from rotated coordinates back to non-rotated image.

    Returns
    -------
    rotated_point : tuple
        Rotated point in the format of ``(x, y)`` as measured from origin.

    """
    #  unpack the image and rotated images shapes
    if reverse:
        angle = (angle * -1)
        temp = ishape
        ishape = rshape
        rshape = temp

    # transform into center of image coordinates
    yhalf, xhalf = ishape
    yrhalf, xrhalf = rshape

    yhalf = yhalf / 2
    xhalf = xhalf / 2
    yrhalf = yrhalf / 2
    xrhalf = xrhalf / 2

    startx = point[0] - xhalf
    starty = point[1] - yhalf

    # do the rotation
    newx = startx * np.cos(angle) - starty * np.sin(angle)
    newy = startx * np.sin(angle) + starty * np.cos(angle)

    # add back the padding from changing the size of the image
    newx = newx + xrhalf
    newy = newy + yrhalf

    return (newx, newy)