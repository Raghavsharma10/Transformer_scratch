def point_type(name, fields, srid_map):
    """ Dynamically create a Point subclass.
    """

    def srid(self):
        try:
            return srid_map[len(self)]
        except KeyError:
            return None

    attributes = {"srid": property(srid)}

    for index, subclass_field in enumerate(fields):

        def accessor(self, i=index, f=subclass_field):
            try:
                return self[i]
            except IndexError:
                raise AttributeError(f)

        for field_alias in {subclass_field, "xyz"[index]}:
            attributes[field_alias] = property(accessor)

    cls = type(name, (Point,), attributes)

    with __srid_table_lock:
        for dim, srid in srid_map.items():
            __srid_table[srid] = (cls, dim)

    return cls