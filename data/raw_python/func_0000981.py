def validate(payload, schema):
    """Validate `payload` against `schema`, returning an error list.

    jsonschema provides lots of information in it's errors, but it can be a bit
    of work to extract all the information.
    """
    v = jsonschema.Draft4Validator(
        schema, format_checker=jsonschema.FormatChecker())
    error_list = []
    for error in v.iter_errors(payload):
        message = error.message
        location = '/' + '/'.join([str(c) for c in error.absolute_path])
        error_list.append(message + ' at ' + location)
    return error_list