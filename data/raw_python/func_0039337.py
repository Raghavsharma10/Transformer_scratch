def autohard(equation):
    """
    Automatically solve a hard maths problem.

    :type equation: string
    :param equation: The equation to solve.

    >>> autohard("log 10")
    2.302585092994046
    """

    try:
        # Try to set a variable to an integer
        num1 = int(equation.split(" ")[1])

    except ValueError:
        # Try to set a variable to a decimal
        num1 = float(equation.split(" ")[1])

    # If the lowercase version of the operation equals 'log'
    if equation.split(" ")[0].lower() == "log":
        # Return the answer
        return math.log(num1)

    # If the lowercase version of the operation equals 'acos'
    elif equation.split(" ")[0].lower() == "acos":
        # Return the answer
        return math.acos(num1)

    # If the lowercase version of the operation equals 'asin'
    elif equation.split(" ")[0].lower() == "asin":
        # Return the answer
        return math.asin(num1)

    # If the lowercase version of the operation equals 'atan'
    elif equation.split(" ")[0].lower() == "atan":
        # Return the answer
        return math.atan(num1)

    # If the lowercase version of the operation equals 'cos'
    elif equation.split(" ")[0].lower() == "cos":
        # Return the answer
        return math.cos(num1)

    # If the lowercase version of the operation equals 'hypot'
    elif equation.split(" ")[0].lower() == "hypot":
        try:
            # Try to set a variable to an integer
            num2 = int(equation.split(" ")[2])

        except ValueError:
            # Try to set a variable to an decimal
            num2 = float(equation.split(" ")[2])

        # Return the answer
        return math.hypot(num1, num2)

    # If the lowercase version of the operation equals 'sin'
    elif equation.split(" ")[0].lower() == "sin":
        # Return the answer
        return math.sin(num1)

    # If the lowercase version of the operation equals 'tan'
    elif equation.split(" ")[0].lower() == "tan":
        # Return the answer
        return math.tan(num1)

    # Raise a warning
    raise ValueError("Invalid operation entered.")