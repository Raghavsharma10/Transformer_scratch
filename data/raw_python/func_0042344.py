def wrap_function(func):
    """
    RETURN A THREE-PARAMETER WINDOW FUNCTION TO MATCH
    """
    if is_text(func):
        return compile_expression(func)

    numarg = func.__code__.co_argcount
    if numarg == 0:

        def temp(row, rownum, rows):
            return func()

        return temp
    elif numarg == 1:

        def temp(row, rownum, rows):
            return func(row)

        return temp
    elif numarg == 2:

        def temp(row, rownum, rows):
            return func(row, rownum)

        return temp
    elif numarg == 3:
        return func