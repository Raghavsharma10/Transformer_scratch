def randomnum(minimum=1, maximum=2, seed=None):
    """
    Generate a random number.

    :type minimum: integer
    :param minimum: The minimum number to generate.

    :type maximum: integer
    :param maximum: The maximum number to generate.

    :type seed: integer
    :param seed: A seed to use when generating the random number.

    :return: The randomized number.
    :rtype: integer

    :raises TypeError: Minimum number is not a number.
    :raises TypeError: Maximum number is not a number.

    >>> randomnum(1, 100, 150)
    42
    """

    if not (isnum(minimum)):
        raise TypeError("Minimum number is not a number.")

    if not (isnum(maximum)):
        raise TypeError("Maximum number is not a number.")

    if seed is None:
        return random.randint(minimum, maximum)

    random.seed(seed)
    return random.randint(minimum, maximum)