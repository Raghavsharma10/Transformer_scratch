def LazyField(lookup_name, scope):
    """Super non-standard stuff here. Dynamically changing the base
    class using the scope and the lazy name when the class is
    instantiated. This works as long as the original base class is
    not directly inheriting from object (which we're not, since
    our original base class is fields.Field).
    """
    def __init__(self, stream=None):
        base_cls = self._pfp__scope.get_id(self._pfp__lazy_name)
        self.__class__.__bases__ = (base_cls,)
        base_cls.__init__(self, stream)

    new_class = type(lookup_name + "_lazy", (fields.Field,), {
        "__init__"            : __init__,
        "_pfp__scope"        : scope,
        "_pfp__lazy_name"    : lookup_name
    })
    return new_class