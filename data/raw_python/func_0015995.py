def validate_string(string, options=None):
    """Validate the input `string` according to the options passed in.

    If any exceptions are raised during validation, no further validation
    will take place.

    Args:
        string: The string containing the JSON to be validated.
        options: An instance of ``ValidationOptions``.

    Returns:
        An ObjectValidationResults instance, or a list of such.

    """
    output.info("Performing JSON schema validation on input string: " + string)
    stream = io.StringIO(string)
    return validate(stream, options)