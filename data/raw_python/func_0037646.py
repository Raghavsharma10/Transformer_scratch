def prepare_sql(sql, add_semicolon=True, invalid_starts=('--', '/*', '*/', ';')):
    """Wrapper method for PrepareSQL class."""
    return PrepareSQL(sql, add_semicolon, invalid_starts).prepared