def Cross(width=3, color=0):
    """Draws a cross centered in the target area

    :param width: width of the lines of the cross in pixels
    :type width: int
    :param color: color of the lines of the cross
    :type color: pygame.Color
    """
    return Overlay(Line("h", width, color), Line("v", width, color))