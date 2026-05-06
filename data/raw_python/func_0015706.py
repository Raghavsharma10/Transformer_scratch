def list_properties(type):
    """
    :param type: a Python GObject instance or type that the signal is associated with
    :type type: :obj:`GObject.Object`

    :returns: a list of :obj:`GObject.ParamSpec`
    :rtype: [:obj:`GObject.ParamSpec`]

    Takes a GObject/GInterface subclass or a GType and returns a list of
    GParamSpecs for all properties of `type`.
    """

    if isinstance(type, PGType):
        type = type.pytype

    from pgi.obj import Object, InterfaceBase

    if not issubclass(type, (Object, InterfaceBase)):
        raise TypeError("Must be a subclass of %s or %s" %
                        (Object.__name__, InterfaceBase.__name__))

    gparams = []
    for key in dir(type.props):
        if not key.startswith("_"):
            gparams.append(getattr(type.props, key))
    return gparams