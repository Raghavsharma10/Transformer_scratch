def resolve_schema(schema):
    """Transform JSON schemas "allOf".

    This is the default schema resolver.

    This function was created because some javascript JSON Schema libraries
    don't support "allOf". We recommend to use this function only in this
    specific case.

    This function is transforming the JSON Schema by removing "allOf" keywords.
    It recursively merges the sub-schemas as dictionaries. The process is
    completely custom and works only for simple JSON Schemas which use basic
    types (object, string, number, ...). Optional structures like "schema
    dependencies" or "oneOf" keywords are not supported.

    :param dict schema: the schema to resolve.
    :returns: the resolved schema

    .. note::

        The schema should have the ``$ref`` already resolved before running
        this method.
    """
    def traverse(schema):
        if isinstance(schema, dict):
            if 'allOf' in schema:
                for x in schema['allOf']:
                    sub_schema = x
                    sub_schema.pop('title', None)
                    schema = _merge_dicts(schema, sub_schema)
                schema.pop('allOf')
                schema = traverse(schema)
            elif 'properties' in schema:
                for x in schema.get('properties', []):
                    schema['properties'][x] = traverse(
                        schema['properties'][x])
            elif 'items' in schema:
                schema['items'] = traverse(schema['items'])
        return schema
    return traverse(schema)