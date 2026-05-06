def find_field(ctx, search, by_type, obj):
    """Find fields in registered data models."""

    # TODO: Fix this to work recursively on all possible subschemes
    if search is not None:
        search = search
    else:
        search = _ask("Enter search term")

    database = ctx.obj['db']

    def find(search_schema, search_field, find_result=None, key=""):
        """Examine a schema to find fields by type or name"""

        if find_result is None:
            find_result = []
        fields = search_schema['properties']
        if not by_type:
            if search_field in fields:
                find_result.append(key)
                # log("Found queried fieldname in ", model)
        else:
            for field in fields:
                try:
                    if "type" in fields[field]:
                        # log(fields[field], field)
                        if fields[field]["type"] == search_field:
                            find_result.append((key, field))
                            # log("Found field", field, "in", model)
                except KeyError as e:
                    log("Field access error:", e, type(e), exc=True,
                        lvl=debug)

        if 'properties' in fields:
            # log('Sub properties checking:', fields['properties'])
            find_result.append(find(fields['properties'], search_field,
                                    find_result, key=fields['name']))

        for field in fields:
            if 'items' in fields[field]:
                if 'properties' in fields[field]['items']:
                    # log('Sub items checking:', fields[field])
                    find_result.append(find(fields[field]['items'], search_field,
                                            find_result, key=field))
                else:
                    pass
                    # log('Items without proper definition!')

        return find_result

    if obj is not None:
        schema = database.objectmodels[obj]._schema
        result = find(schema, search, [], key="top")
        if result:
            # log(args.object, result)
            print(obj)
            pprint(result)
    else:
        for model, thing in database.objectmodels.items():
            schema = thing._schema

            result = find(schema, search, [], key="top")
            if result:
                print(model)
                # log(model, result)
                print(result)