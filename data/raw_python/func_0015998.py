def load_schema(schema_path):
    """Load the JSON schema at the given path as a Python object.

    Args:
        schema_path: A filename for a JSON schema.

    Returns:
        A Python object representation of the schema.

    """
    try:
        with open(schema_path) as schema_file:
            schema = json.load(schema_file)
    except ValueError as e:
        raise SchemaInvalidError('Invalid JSON in schema or included schema: '
                                 '%s\n%s' % (schema_file.name, str(e)))

    return schema