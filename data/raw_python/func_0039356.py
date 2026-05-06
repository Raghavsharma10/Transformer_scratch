def lcm(num1, num2):
    """
    Find the lowest common multiple of 2 numbers

    :type num1: number
    :param num1: The first number to find the lcm for

    :type num2: number
    :param num2: The second number to find the lcm for
    """

    if num1 > num2:
        bigger = num1
    else:
        bigger = num2
    while True:
        if bigger % num1 == 0 and bigger % num2 == 0:
            return bigger
        bigger += 1