def equation(operation, firstnum, secondnum):
    """
    Solve a simple maths equation manually
    """
    if operation == 'plus':
        return firstnum + secondnum
    elif operation == 'minus':
        return firstnum - secondnum
    elif operation == 'multiply':
        return firstnum * secondnum
    elif operation == 'divide':
        if not secondnum == 0:
            return firstnum / secondnum
        raise ZeroDivisionError("Unable to divide by 0.")
    raise ValueError('Invalid operation provided.')