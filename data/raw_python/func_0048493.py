def deprecated(operation=None):
    """
    Mark an operation deprecated.
    """
    def inner(o):
        o.deprecated = True
        return o
    return inner(operation) if operation else inner