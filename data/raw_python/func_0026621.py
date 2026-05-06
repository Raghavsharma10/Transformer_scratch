def Group(expressions, final_function, inbetweens, name=""):
    """ Group expressions together with ``inbetweens`` and with the output of a ``final_functions``.
    """
    lengths = []
    functions = []
    regex = ""
    i = 0
    for expression in expressions:
        regex += inbetweens[i]
        regex += "(?:" + expression.regex + ")"
        lengths.append(sum(expression.group_lengths))
        functions.append(expression.run)
        i += 1
    regex += inbetweens[i]

    return Expression(regex, functions, lengths, final_function, name)