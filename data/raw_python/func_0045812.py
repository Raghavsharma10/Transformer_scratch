def security(name, *permissions):
    """
    Decorator to add security definition.
    """
    def inner(c):
        c.security = Security(name, *permissions)
        return c
    return inner