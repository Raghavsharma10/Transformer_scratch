def factors(number):
    """
    Find all of the factors of a number and return it as a list.

    :type number: integer
    :param number: The number to find the factors for.
    """

    if not (isinstance(number, int)):
        raise TypeError(
            "Incorrect number type provided. Only integers are accepted.")

    factors = []
    for i in range(1, number + 1):
        if number % i == 0:
            factors.append(i)
    return factors