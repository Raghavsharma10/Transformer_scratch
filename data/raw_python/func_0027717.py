def upgradeSystemOid(store):
    """
    Upgrade the system tables to use explicit oid columns.
    """
    store.transact(
        _upgradeTableOid, store, 'axiom_types',
        lambda: store.executeSchemaSQL(CREATE_TYPES))
    store.transact(
        _upgradeTableOid, store, 'axiom_objects',
        lambda: store.executeSchemaSQL(CREATE_OBJECTS),
        lambda: store.executeSchemaSQL(CREATE_OBJECTS_IDX))