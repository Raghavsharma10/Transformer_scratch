def _upgradeTableOid(store, table, createTable, postCreate=lambda: None):
    """
    Upgrade a table to have an explicit oid.

    Must be called in a transaction to avoid corrupting the database.
    """
    if _hasExplicitOid(store, table):
        return
    store.executeSchemaSQL(
        'ALTER TABLE *DATABASE*.{0} RENAME TO {0}_temp'.format(table))
    createTable()
    store.executeSchemaSQL(
        'INSERT INTO *DATABASE*.{0} '
        'SELECT oid, * FROM *DATABASE*.{0}_temp'.format(table))
    store.executeSchemaSQL('DROP TABLE *DATABASE*.{0}_temp'.format(table))
    postCreate()