def delete(ctx, schema, uuid, object_filter, yes):
    """Delete stored objects (CAUTION!)"""

    database = ctx.obj['db']

    if schema is None:
        log('No schema given. Read the help', lvl=warn)
        return

    model = database.objectmodels[schema]

    if uuid:
        count = model.count({'uuid': uuid})
        obj = model.find({'uuid': uuid})
    elif object_filter:
        count = model.count(literal_eval(object_filter))
        obj = model.find(literal_eval(object_filter))
    else:
        count = model.count()
        obj = model.find()

    if count == 0:
        log('No objects to delete found')
        return

    if not yes and not _ask("Are you sure you want to delete %i objects" % count,
                            default=False, data_type="bool", show_hint=True):
        return

    for item in obj:
        item.delete()

    log('Done')