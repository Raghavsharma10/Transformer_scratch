def gcd(vector):
    """
    Calculate the greatest common divisor (GCD) of a sequence of numbers.

    The sequence can be a list of numbers or a Numpy vector of numbers. The
    computations are carried out with a precision of 1E-12 if the objects are
    not `fractions <https://docs.python.org/3/library/fractions.html>`_. When
    possible it is best to use the `fractions
    <https://docs.python.org/3/library/fractions.html>`_ data type with the
    numerator and denominator arguments when computing the GCD of floating
    point numbers.

    :param vector: Vector of numbers
    :type  vector: list of numbers or Numpy vector of numbers
    """
    # pylint: disable=C1801
    if not len(vector):
        return None
    if len(vector) == 1:
        return vector[0]
    if len(vector) == 2:
        return pgcd(vector[0], vector[1])
    current_gcd = pgcd(vector[0], vector[1])
    for element in vector[2:]:
        current_gcd = pgcd(current_gcd, element)
    return current_gcd