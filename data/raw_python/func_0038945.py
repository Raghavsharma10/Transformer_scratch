def getvar(root, name, vtype='', dimensions=(), digits=0, fill_value=None,
           source=None):
    """
    Return a variable from a NCFile or NCPackage instance. If the variable
    doesn't exists create it.

    Keyword arguments:
    root -- the root descriptor returned by the 'open' function
    name -- the name of the variable
    vtype -- the type of each value, ex ['f4', 'i4', 'i1', 'S1'] (default '')
    dimensions -- the tuple with dimensions name of the variables (default ())
    digits -- the precision required when using a 'f4' vtype (default 0)
    fill_value -- the initial value used in the creation time (default None)
    source -- the source variable to be copied (default None)
    """
    return root.getvar(name, vtype, dimensions, digits, fill_value, source)