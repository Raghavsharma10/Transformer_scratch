def convert_schema(raml_schema, mime_type):
    """ Restructure `raml_schema` to a dictionary that has 'properties'
    as well as other schema keys/values.

    The resulting dictionary looks like this::

    {
        "properties": {
            "field1": {
                "required": boolean,
                "type": ...,
                ...more field options
            },
            ...more properties
        },
        "public_fields": [...],
        "auth_fields": [...],
        ...more schema options
    }

    :param raml_schema: RAML request body schema.
    :param mime_type: ContentType of the schema as a string from RAML
        file. Only JSON is currently supported.
    """
    if mime_type == ContentTypes.JSON:
        if not isinstance(raml_schema, dict):
            raise TypeError(
                'Schema is not a valid JSON. Please check your '
                'schema syntax.\n{}...'.format(str(raml_schema)[:60]))
        return raml_schema
    if mime_type == ContentTypes.TEXT_XML:
        # Process XML schema
        pass