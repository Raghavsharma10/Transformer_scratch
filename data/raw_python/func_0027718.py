def upgradeExplicitOid(store):
    """
    Upgrade a store to use explicit oid columns.
    
    This allows VACUUMing the database without corrupting it.

    This requires copying all of axiom_objects and axiom_types, as well as all
    item tables that have not yet been upgraded.  Consider VACUUMing the
    database afterwards to reclaim space.
    """
    upgradeSystemOid(store)
    for typename, version in store.querySchemaSQL(LATEST_TYPES):
        cls = _typeNameToMostRecentClass[typename]
        if cls.schemaVersion != version:
            remaining = store.querySQL(
                'SELECT oid FROM {} LIMIT 1'.format(
                    store._tableNameFor(typename, version)))
            if len(remaining) == 0:
                # Nothing to upgrade
                continue
            else:
                raise RuntimeError(
                    '{}:{} not fully upgraded to {}'.format(
                        typename, version, cls.schemaVersion))
        store.transact(
            _upgradeTableOid,
            store,
            store._tableNameOnlyFor(typename, version),
            lambda: store._justCreateTable(cls),
            lambda: store._createIndexesFor(cls, []))