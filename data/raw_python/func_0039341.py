def circleconvert(amount, currentformat, newformat):
    """
    Convert a circle measurement.

    :type amount: number
    :param amount: The number to convert.

    :type currentformat: string
    :param currentformat: The format of the provided value.

    :type newformat: string
    :param newformat: The intended format of the value.

    >>> circleconvert(45, "radius", "diameter")
    90
    """

    # If the same format was provided
    if currentformat.lower() == newformat.lower():
        # Return the provided value
        return amount

    # If the lowercase version of the current format is 'radius'
    if currentformat.lower() == 'radius':
        # If the lowercase version of the new format is 'diameter'
        if newformat.lower() == 'diameter':
            # Return the converted value
            return amount * 2

        # If the lowercase version of the new format is 'circumference'
        elif newformat.lower() == 'circumference':
            # Return the converted value
            return amount * 2 * math.pi

        # Raise a warning
        raise ValueError("Invalid new format provided.")

    # If the lowercase version of the current format is 'diameter'
    elif currentformat.lower() == 'diameter':
        # If the lowercase version of the new format is 'radius'
        if newformat.lower() == 'radius':
            # Return the converted value
            return amount / 2

        # If the lowercase version of the new format is 'circumference'
        elif newformat.lower() == 'circumference':
            # Return the converted value
            return amount * math.pi

        # Raise a warning
        raise ValueError("Invalid new format provided.")

    # If the lowercase version of the current format is 'circumference'
    elif currentformat.lower() == 'circumference':
        # If the lowercase version of the new format is 'radius'
        if newformat.lower() == 'radius':
            # Return the converted value
            return amount / math.pi / 2

        # If the lowercase version of the new format is 'diameter'
        elif newformat.lower() == 'diameter':
            # Return the converted value
            return amount / math.pi