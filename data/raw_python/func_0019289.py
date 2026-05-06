def classname(self):
    """Return the class name of the given instance object or class.

    >>> from hydpy.core.objecttools import classname
    >>> from hydpy import pub
    >>> print(classname(float))
    float
    >>> print(classname(pub.options))
    Options
    """
    if inspect.isclass(self):
        string = str(self)
    else:
        string = str(type(self))
    try:
        string = string.split("'")[1]
    except IndexError:
        pass
    return string.split('.')[-1]