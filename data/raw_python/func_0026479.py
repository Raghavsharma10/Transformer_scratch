def db_import(ctx, schema, uuid, object_filter, import_format, filename, all_schemata, dry):
    """Import objects from file

    Warning! This functionality is work in progress and you may destroy live data by using it!
    Be very careful when using the export/import functionality!"""

    import_format = import_format.upper()

    with open(filename, 'r') as f:
        json_data = f.read()
    data = json.loads(json_data)  # , parse_float=True, parse_int=True)

    if schema is None:
        if all_schemata is False:
            log('No schema given. Read the help', lvl=warn)
            return
        else:
            schemata = data.keys()
    else:
        schemata = [schema]

    from hfos import database
    database.initialize(ctx.obj['dbhost'], ctx.obj['dbname'])

    all_items = {}
    total = 0

    for schema_item in schemata:
        model = database.objectmodels[schema_item]

        objects = data[schema_item]
        if uuid:
            for item in objects:
                if item['uuid'] == uuid:
                    items = [model(item)]
        else:
            items = []
            for item in objects:
                thing = model(item)
                items.append(thing)

        schema_total = len(items)
        total += schema_total

        if dry:
            log('Would import', schema_total, 'items of', schema_item)
        all_items[schema_item] = items

    if dry:
        log('Would import', total, 'objects.')
    else:
        log('Importing', total, 'objects.')
        for schema_name, item_list in all_items.items():
            log('Importing', len(item_list), 'objects of type', schema_name)
            for item in item_list:
                item._fields['_id'] = bson.objectid.ObjectId(item._fields['_id'])
                item.save()