def _build_collections(store):
    """Generate database collections with indices from the schemastore"""

    result = {}

    client = pymongo.MongoClient(host=dbhost, port=dbport)
    db = client[dbname]

    for schemaname in store:

        schema = None
        indices = None

        try:
            schema = store[schemaname]['schema']
            indices = store[schemaname].get('indices', None)
        except KeyError:
            db_log("No schema found for ", schemaname, lvl=critical)

        try:
            result[schemaname] = db[schemaname]
        except Exception:
            db_log("Could not get collection for schema ", schemaname, schema, lvl=critical, exc=True)

        if indices is not None:
            col = db[schemaname]
            db_log('Adding indices to', schemaname, lvl=debug)
            i = 0
            keys = list(indices.keys())

            while i < len(indices):
                index_name = keys[i]
                index = indices[index_name]

                index_type = index.get('type', None)
                index_unique = index.get('unique', False)
                index_sparse = index.get('sparse', True)
                index_reindex = index.get('reindex', False)

                if index_type in (None, 'text'):
                    index_type = pymongo.TEXT
                elif index_type == '2dsphere':
                    index_type = pymongo.GEOSPHERE

                def do_index():
                    col.ensure_index([(index_name, index_type)],
                                     unique=index_unique,
                                     sparse=index_sparse)

                db_log('Enabling index of type', index_type, 'on', index_name, lvl=debug)
                try:
                    do_index()
                    i += 1
                except pymongo.errors.OperationFailure:
                    db_log(col.list_indexes().__dict__, pretty=True, lvl=verbose)
                    if not index_reindex:
                        db_log('Index was not created!', lvl=warn)
                        i += 1
                    else:
                        try:
                            col.drop_index(index_name)
                            do_index()
                            i += 1
                        except pymongo.errors.OperationFailure as e:
                            db_log('Index recreation problem:', exc=True, lvl=error)
                            col.drop_indexes()
                            i = 0

                            # for index in col.list_indexes():
                            #    db_log("Index: ", index)
    return result