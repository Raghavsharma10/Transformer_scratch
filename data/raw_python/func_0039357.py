def hcf(num1, num2):
    """
    Find the highest common factor of 2 numbers

    :type num1: number
    :param num1: The first number to find the hcf for

    :type num2: number
    :param num2: The second number to find the hcf for
    """

    if num1 > num2:
        smaller = num2
    else:
        smaller = num1
    for i in range(1, smaller + 1):
        if ((num1 % i == 0) and (num2 % i == 0)):
            return i