def new_class_from_gtype(gtype):
    """Create a new class for a gtype not in the gir.
    The caller is responsible for caching etc.
    """

    if gtype.is_a(PGType.from_name("GObject")):
        parent = gtype.parent.pytype
        if parent is None or parent == PGType.from_name("void"):
            return
        interfaces = [i.pytype for i in gtype.interfaces]
        bases = tuple([parent] + interfaces)

        cls = type(gtype.name, bases, dict())
        cls.__gtype__ = gtype

        return cls
    elif gtype.is_a(PGType.from_name("GEnum")):
        from pgi.enum import GEnumBase
        return GEnumBase