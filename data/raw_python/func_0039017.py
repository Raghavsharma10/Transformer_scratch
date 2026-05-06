def debug(value):
    """
        Simple tag to debug output a variable;

        Usage:
            {% debug request %}
    """
    print("%s %s: " % (type(value), value))
    print(dir(value))
    print('\n\n')
    return ''