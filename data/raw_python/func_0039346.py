def posnegtoggle(number):
    """
    Toggle a number between positive and negative.
    The converter works as follows:

    - 1 > -1
    - -1 > 1
    - 0 > 0

    :type number: number
    :param number: The number to toggle.
    """
    if bool(number > 0):
        return number - number * 2
    elif bool(number < 0):
        return number + abs(number) * 2
    elif bool(number == 0):
        return number