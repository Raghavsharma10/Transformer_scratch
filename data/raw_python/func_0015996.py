def load_validator(schema_path, schema):
    """Create a JSON schema validator for the given schema.

    Args:
        schema_path: The filename of the JSON schema.
        schema: A Python object representation of the same schema.

    Returns:
        An instance of Draft4Validator.

    """
    # Get correct prefix based on OS
    if os.name == 'nt':
        file_prefix = 'file:///'
    else:
        file_prefix = 'file:'

    resolver = RefResolver(file_prefix + schema_path.replace("\\", "/"), schema)
    validator = Draft4Validator(schema, resolver=resolver)

    return validator