def get_linenumbers(functions, module, searchstr='def {}(image):\n'):
    """Returns a dictionary which maps function names to line numbers.

    Args:
        functions: a list of function names
        module:    the module to look the functions up
        searchstr: the string to search for
    Returns:
        A dictionary with functions as keys and their line numbers as values.
    """
    lines = inspect.getsourcelines(module)[0]
    line_numbers = {}
    for function in functions:
        try:
            line_numbers[function] = lines.index(
                    searchstr.format(function)) + 1
        except ValueError:
            print(r'Can not find `{}`'.format(searchstr.format(function)))
            line_numbers[function] = 0
    return line_numbers