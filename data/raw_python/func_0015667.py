def get_foreign_struct(namespace, name):
    """Returns a ForeignStruct implementation or raises ForeignError"""

    get_foreign_module(namespace)

    try:
        return ForeignStruct.get(namespace, name)
    except KeyError:
        raise ForeignError("Foreign %s.%s not supported" % (namespace, name))