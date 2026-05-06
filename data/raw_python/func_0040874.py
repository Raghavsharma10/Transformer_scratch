def is_modified(row, dialect):
    """
    Has the row data been modified?

    This method inspects the row, and iterates over all columns looking for changes
    to the (processed) data, skipping over unmodified columns.

    :param row: SQLAlchemy model instance
    :param dialect: :py:class:`~sqlalchemy.engine.interfaces.Dialect`
    :return: True if any columns were modified, else False
    """
    ins = inspect(row)
    modified_cols = set(get_column_keys(ins.mapper)) - ins.unmodified
    for col_name in modified_cols:
        current_value = get_column_attribute(row, col_name, dialect=dialect)
        previous_value = get_column_attribute(row, col_name, use_dirty=False, dialect=dialect)
        if previous_value != current_value:
            return True
    return False