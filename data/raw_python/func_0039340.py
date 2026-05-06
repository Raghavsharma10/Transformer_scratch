def fracsimplify(numerator, denominator):
    """
    Simplify a fraction.

    :type numerator: integer
    :param numerator: The numerator of the fraction to simplify

    :type denominator: integer
    :param denominator: The denominator of the fraction to simplify

    :return: The simplified fraction
    :rtype: list
    """

    # If the numerator is the same as the denominator
    if numerator == denominator:
        # Return the most simplified fraction
        return '1/1'

    # If the numerator is larger than the denominator
    elif int(numerator) > int(denominator):
        # Set the limit to half of the numerator
        limit = int(numerator / 2)

    elif int(numerator) < int(denominator):

        # Set the limit to half of the denominator
        limit = int(denominator / 2)

    # For each item in range from 2 to the limit
    for i in range(2, limit):
        # Set the number to check as the limit minus i
        checknum = limit - i
        # If the number is divisible by the numerator and denominator
        if numerator % checknum == 0 and denominator % checknum == 0:
            # Set the numerator to half of the number
            numerator = numerator / checknum
            # Set the denominator to half of the number
            denominator = denominator / checknum

    # Return the integer version of the numerator and denominator
    return [int(numerator), int(denominator)]