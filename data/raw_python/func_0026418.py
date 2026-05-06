def _build_model_factories(store):
    """Generate factories to construct objects from schemata"""

    result = {}

    for schemaname in store:

        schema = None

        try:
            schema = store[schemaname]['schema']
        except KeyError:
            schemata_log("No schema found for ", schemaname, lvl=critical, exc=True)

        try:
            result[schemaname] = warmongo.model_factory(schema)
        except Exception as e:
            schemata_log("Could not create factory for schema ", schemaname, schema, lvl=critical, exc=True)

    return result