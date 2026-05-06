def amountdiv(number, minnum, maxnum):
    """
    Get the amount of numbers divisable by a number.

    :type number: number
    :param number: The number to use.

    :type minnum: integer
    :param minnum: The minimum number to check.

    :type maxnum: integer
    :param maxnum: The maximum number to check.

    >>> amountdiv(20, 1, 15)
    5
    """

    # Set the amount to 0
    amount = 0

    # For each item in range of minimum and maximum
    for i in range(minnum, maxnum + 1):
        # If the remainder of the divided number is 0
        if number % i == 0:
            # Add 1 to the total amount
            amount += 1

    # Return the result
    return amount