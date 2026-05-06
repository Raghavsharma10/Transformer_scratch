def autosolve(equation):
    """
    Automatically solve an easy maths problem.

    :type equation: string
    :param equation: The equation to calculate.

    >>> autosolve("300 + 600")
    900
    """

    try:
        # Try to set a variable to an integer
        num1 = int(equation.split(" ")[0])

    except ValueError:
        # Try to set a variable to a decimal
        num1 = float(equation.split(" ")[0])

    try:
        # Try to set a variable to an integer
        num2 = int(equation.split(" ")[2])

    except ValueError:
        # Try to set a variable to a decimal
        num2 = float(equation.split(" ")[2])

    # If the lowercase version of the operator is '+', 'plus' or 'add'
    if equation.split(" ")[1].lower() in ["+", "plus", "add"]:

        # Return the answer
        return num1 + num2

    # If the lowercase version of the operator is '-', 'minus' or 'subtract'
    elif equation.split(" ")[1].lower() in ["-", "minus", "subtract"]:

        # Return the answer
        return num1 - num2

    # If the lowercase version of the operator is '*', 'times', 'multiply'
    elif equation.split(" ")[1].lower() in ["*", "times", "multiply"]:

        # Return the answer
        return num1 * num2

    # If the lowercase version of the operator is '/', 'divide' or 'quotient'
    elif equation.split(" ")[1].lower() in ["/", "divide", "quotient"]:

        # Return the answer
        return num1 / num2

    # If the lowercase version of the operator is '%, 'remainder' or 'rem'
    elif equation.split(" ")[1].lower() in ["%", "remainder", "rem"]:

        # Return the answer
        return num1 % num2

    # Raise a warning
    raise ValueError("Invalid operation provided.")