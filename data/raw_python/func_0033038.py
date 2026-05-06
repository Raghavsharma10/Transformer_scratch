def db_type(python_type_representation):
    """
    Converts from any of:
        (1) Python type
        (2) NumPy/Pandas dtypes
        (3) string names of types
    ...to a sqlite3 type name
    """
    for type_name in _candidate_type_names(python_type_representation):
        db_type_name = _lookup_type_name(type_name)
        if db_type_name:
            return db_type_name
    raise ValueError("Failed to find sqlite3 column type for %s" % (
        python_type_representation))