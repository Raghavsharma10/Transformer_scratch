def clear(ctx, schema):
    """Clears an entire database collection irrevocably. Use with caution!"""

    response = _ask('Are you sure you want to delete the collection "%s"' % (
        schema), default='N', data_type='bool')
    if response is True:
        host, port = ctx.obj['dbhost'].split(':')

        client = pymongo.MongoClient(host=host, port=int(port))
        database = client[ctx.obj['dbname']]

        log("Clearing collection for", schema, lvl=warn,
            emitter='MANAGE')
        result = database.drop_collection(schema)
        if not result['ok']:
            log("Could not drop collection:", lvl=error)
            log(result, pretty=True, lvl=error)
        else:
            log("Done")