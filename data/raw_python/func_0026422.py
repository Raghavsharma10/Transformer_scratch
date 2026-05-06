def backup(schema, uuid, export_filter, export_format, filename, pretty, export_all, omit):
    """Exports all collections to (JSON-) files."""

    export_format = export_format.upper()

    if pretty:
        indent = 4
    else:
        indent = 0

    f = None

    if filename:
        try:
            f = open(filename, 'w')
        except (IOError, PermissionError) as e:
            backup_log('Could not open output file for writing:', exc=True, lvl=error)
            return

    def output(what, convert=False):
        """Output the backup in a specified format."""

        if convert:
            if export_format == 'JSON':
                data = json.dumps(what, indent=indent)
            else:
                data = ""
        else:
            data = what

        if not filename:
            print(data)
        else:
            f.write(data)

    if schema is None:
        if export_all is False:
            backup_log('No schema given.', lvl=warn)
            return
        else:
            schemata = objectmodels.keys()
    else:
        schemata = [schema]

    all_items = {}

    for schema_item in schemata:
        model = objectmodels[schema_item]

        if uuid:
            obj = model.find({'uuid': uuid})
        elif export_filter:
            obj = model.find(literal_eval(export_filter))
        else:
            obj = model.find()

        items = []
        for item in obj:
            fields = item.serializablefields()
            for field in omit:
                try:
                    fields.pop(field)
                except KeyError:
                    pass
            items.append(fields)

        all_items[schema_item] = items

        # if pretty is True:
        #    output('\n// Objectmodel: ' + schema_item + '\n\n')
        # output(schema_item + ' = [\n')

    output(all_items, convert=True)

    if f is not None:
        f.flush()
        f.close()