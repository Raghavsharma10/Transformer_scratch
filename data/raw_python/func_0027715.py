def _hasExplicitOid(store, table):
    """
    Does the given table have an explicit oid column?
    """
    return any(info[1] == 'oid' for info
               in store.querySchemaSQL(
                   'PRAGMA *DATABASE*.table_info({})'.format(table)))