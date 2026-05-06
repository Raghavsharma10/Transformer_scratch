def name(self):
    """Name of the class of the given instance in lower case letters.

    This function is thought to be implemented as a property.  Otherwise
    it would violate the principle not to access or manipulate private
    attributes ("_name"):

    >>> from hydpy.core.objecttools import name
    >>> class Test(object):
    ...     name = property(name)
    >>> test1 = Test()
    >>> test1.name
    'test'
    >>> test1._name
    'test'

    The private attribute is added for performance reasons only.  Note that
    it is a class attribute:

    >>> test2 = Test()
    >>> test2._name
    'test'
    """
    cls = type(self)
    try:
        return cls.__dict__['_name']
    except KeyError:
        setattr(cls, '_name', instancename(self))
        return cls.__dict__['_name']