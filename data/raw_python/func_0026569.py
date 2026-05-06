def validate(ctx, schema, all_schemata):
    """Validates all objects or all objects of a given schema."""

    database = ctx.obj['db']

    if schema is None:
        if all_schemata is False:
            log('No schema given. Read the help', lvl=warn)
            return
        else:
            schemata = database.objectmodels.keys()
    else:
        schemata = [schema]

    for schema in schemata:
        try:
            things = database.objectmodels[schema]
            with click.progressbar(things.find(), length=things.count(),
                                   label='Validating %15s' % schema) as object_bar:
                for obj in object_bar:
                    obj.validate()
        except Exception as e:

            log('Exception while validating:',
                schema, e, type(e),
                '\n\nFix this object and rerun validation!',
                emitter='MANAGE', lvl=error)

    log('Done')