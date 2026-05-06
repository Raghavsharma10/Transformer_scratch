def _get_error_generator(type, obj, schema_dir=None, version=DEFAULT_VER, default='core'):
    """Get a generator for validating against the schema for the given object type.

    Args:
        type (str): The object type to find the schema for.
        obj: The object to be validated.
        schema_dir (str): The path in which to search for schemas.
        version (str): The version of the STIX specification to validate
            against. Only used to find base schemas when schema_dir is None.
        default (str): If the schema for the given type cannot be found, use
            the one with this name instead.

    Returns:
        A generator for errors found when validating the object against the
        appropriate schema, or None if schema_dir is None and the schema
        cannot be found.
    """
    # If no schema directory given, use default for the given STIX version,
    # which comes bundled with this package
    if schema_dir is None:
        schema_dir = os.path.abspath(os.path.dirname(__file__) + '/schemas-'
                                     + version + '/')

    try:
        schema_path = find_schema(schema_dir, type)
        schema = load_schema(schema_path)
    except (KeyError, TypeError):
        # Assume a custom object with no schema
        try:
            schema_path = find_schema(schema_dir, default)
            schema = load_schema(schema_path)
        except (KeyError, TypeError):
            # Only raise an error when checking against default schemas, not custom
            if schema_dir is not None:
                return None
            raise SchemaInvalidError("Cannot locate a schema for the object's "
                                     "type, nor the base schema ({}.json).".format(default))

    if type == 'observed-data' and schema_dir is None:
        # Validate against schemas for specific observed data object types later.
        # If schema_dir is not None the schema is custom and won't need to be modified.
        schema['allOf'][1]['properties']['objects'] = {
            "objects": {
                "type": "object",
                "minProperties": 1
            }
        }

    # Don't use custom validator; only check schemas, no additional checks
    validator = load_validator(schema_path, schema)
    try:
        error_gen = validator.iter_errors(obj)
    except schema_exceptions.RefResolutionError:
        raise SchemaInvalidError('Invalid JSON schema: a JSON '
                                 'reference failed to resolve')
    return error_gen