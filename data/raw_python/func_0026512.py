def rbac(ctx, schema, object_filter, action, role, all_schemata):
    """[GROUP] Role based access control"""

    database = ctx.obj['db']

    if schema is None:
        if all_schemata is False:
            log('No schema given. Read the RBAC group help', lvl=warn)
            sys.exit()
        else:
            schemata = database.objectmodels.keys()
    else:
        schemata = [schema]

    things = []

    if object_filter is None:
        parsed_filter = {}
    else:
        parsed_filter = json.loads(object_filter)

    for schema in schemata:
        for obj in database.objectmodels[schema].find(parsed_filter):
            things.append(obj)

    if len(things) == 0:
        log('No objects matched the criteria.', lvl=warn)
        sys.exit()

    ctx.obj['objects'] = things
    ctx.obj['action'] = action
    ctx.obj['role'] = role