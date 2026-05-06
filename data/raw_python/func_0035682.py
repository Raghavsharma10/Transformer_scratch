def _get_col_epsg(mapped_class, geom_attr):
    """Get the EPSG code associated with a geometry attribute.

    Arguments:


    geom_attr
        the key of the geometry property as defined in the SQLAlchemy
        mapper. If you use ``declarative_base`` this is the name of
        the geometry attribute as defined in the mapped class.
    """
    col = class_mapper(mapped_class).get_property(geom_attr).columns[0]
    return col.type.srid