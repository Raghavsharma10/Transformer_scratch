def extend(cls):
    """
    DECORATOR TO ADD METHODS TO CLASSES
    :param cls: THE CLASS TO ADD THE METHOD TO
    :return:
    """
    def extender(func):
        setattr(cls, get_function_name(func), func)
        return func
    return extender