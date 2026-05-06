def sort_schemas(schemas):
    """Sort a list of SQL schemas in order"""
    def keyfun(v):
        x = SQL_SCHEMA_REGEXP.match(v).groups()
        # x3: 'DEV' should come before ''
        return (int(x[0]), x[1], int(x[2]) if x[2] else None,
                x[3] if x[3] else 'zzz', int(x[4]))

    return sorted(schemas, key=keyfun)