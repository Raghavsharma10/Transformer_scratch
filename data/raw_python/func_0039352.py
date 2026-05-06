def isprime(number):
    """
    Check if a number is a prime number

    :type number: integer
    :param number: The number to check
    """

    if number == 1:
        return False
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return False
    return True