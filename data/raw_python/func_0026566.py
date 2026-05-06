def modify(ctx, schema, uuid, object_filter, field, value):
    """Modify field values of objects"""
    database = ctx.obj['db']

    model = database.objectmodels[schema]
    obj = None

    if uuid:
        obj = model.find_one({'uuid': uuid})
    elif object_filter:
        obj = model.find_one(literal_eval(object_filter))
    else:
        log('No object uuid or filter specified.',
            lvl=error)

    if obj is None:
        log('No object found',
            lvl=error)
        return

    log('Object found, modifying', lvl=debug)
    try:
        new_value = literal_eval(value)
    except ValueError:
        log('Interpreting value as string')
        new_value = str(value)

    obj._fields[field] = new_value
    obj.validate()
    log('Changed object validated', lvl=debug)
    obj.save()
    log('Done')