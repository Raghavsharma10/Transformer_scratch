def dir_(self):
    """The prefered way for HydPy objects to respond to |dir|.

    Note the depencence on the `pub.options.dirverbose`.  If this option is
    set `True`, all attributes and methods of the given instance and its
    class (including those inherited from the parent classes) are returned:

    >>> from hydpy import pub
    >>> pub.options.dirverbose = True
    >>> from hydpy.core.objecttools import dir_
    >>> class Test(object):
    ...     only_public_attribute =  None
    >>> print(len(dir_(Test())) > 1) # Long list, try it yourself...
    True

    If the option is set to `False`, only the `public` attributes and methods
    (which do need begin with `_`) are returned:

    >>> pub.options.dirverbose = False
    >>> print(dir_(Test())) # Short list with one single entry...
    ['only_public_attribute']

    If none of those does exists, |dir_| returns a list with a single string
    containing a single empty space (which seems to work better for most
    IDEs than returning an emtpy list):

    >>> del Test.only_public_attribute
    >>> print(dir_(Test()))
    [' ']
    """
    names = set()
    for thing in list(inspect.getmro(type(self))) + [self]:
        for key in vars(thing).keys():
            if hydpy.pub.options.dirverbose or not key.startswith('_'):
                names.add(key)
    if names:
        names = list(names)
    else:
        names = [' ']
    return names