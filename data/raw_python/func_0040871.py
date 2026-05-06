def get_column_attribute(row, col_name, use_dirty=True, dialect=None):
    """
    :param row: the row object
    :param col_name: the column name
    :param use_dirty: whether to return the dirty value of the column
    :param dialect: if not None, should be a :py:class:`~sqlalchemy.engine.interfaces.Dialect`. If \
    specified, this function will process the column attribute into the dialect type before \
    returning it; useful if one is using user defined column types in their mappers.

    :return: if :any:`use_dirty`, this will return the value of col_name on the row before it was \
    changed; else this will return getattr(row, col_name)
    """
    def identity(x):
        return x

    bind_processor = None
    if dialect:
        column_type = getattr(type(row), col_name).type
        bind_processor = get_bind_processor(column_type, dialect)
    bind_processor = bind_processor or identity
    current_value = bind_processor(getattr(row, col_name))
    if use_dirty:
        return current_value

    hist = getattr(inspect(row).attrs, col_name).history
    if not hist.has_changes():
        return current_value
    elif hist.deleted:
        return bind_processor(hist.deleted[0])
    return None