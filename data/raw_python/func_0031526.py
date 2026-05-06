def object_as_dict(obj):
    """Turn an SQLAlchemy model into a dict of field names and values.

    Based on https://stackoverflow.com/a/37350445/1579058
    """
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}