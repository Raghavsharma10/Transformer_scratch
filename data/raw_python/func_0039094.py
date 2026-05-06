def rgb(red, green, blue):
    """
    Calculate the palette index of a color in the 6x6x6 color cube.

    The red, green and blue arguments may range from 0 to 5.
    """
    for value in (red, green, blue):
        if value not in range(6):
            raise ColorError('Value must be within 0-5, was {}.'.format(value))
    return 16 + (red * 36) + (green * 6) + blue