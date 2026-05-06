def get_bind_processor(column_type, dialect):
    """
    Returns a bind processor for a column type and dialect, with special handling
    for JSON/JSONB column types to return dictionaries instead of serialized JSON strings.

    NOTE: This is a workaround for https://github.com/NerdWalletOSS/savage/issues/8

    :param column_type: :py:class:`~sqlalchemy.sql.type_api.TypeEngine`
    :param dialect: :py:class:`~sqlalchemy.engine.interfaces.Dialect`
    :return: bind processor for given column type and dialect
    """
    if column_type.compile(dialect) not in {'JSON', 'JSONB'}:
        # For non-JSON/JSONB column types, return the column type's bind processor
        return column_type.bind_processor(dialect)

    if type(column_type) in {JSON, JSONB}:
        # For bare JSON/JSONB types, we simply skip bind processing altogether
        return None
    elif isinstance(column_type, TypeDecorator) and column_type._has_bind_processor:
        # For decorated JSON/JSONB types, we return the custom bind processor (if any)
        return partial(column_type.process_bind_param, dialect=dialect)
    else:
        # For all other cases, we fall back to deserializing the result of the bind processor
        def wrapped_bind_processor(value):
            json_deserializer = dialect._json_deserializer or json.loads
            return json_deserializer(column_type.bind_processor(dialect)(value))
        return wrapped_bind_processor