def view(ctx, schema, uuid, object_filter):
    """Show stored objects"""

    database = ctx.obj['db']

    if schema is None:
        log('No schema given. Read the help', lvl=warn)
        return

    model = database.objectmodels[schema]

    if uuid:
        obj = model.find({'uuid': uuid})
    elif object_filter:
        obj = model.find(literal_eval(object_filter))
    else:
        obj = model.find()

    for item in obj:
        pprint(item._fields)