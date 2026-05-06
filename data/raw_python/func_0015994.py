def validate_file(fn, options=None):
    """Validate the input document `fn` according to the options passed in.

    If any exceptions are raised during validation, no further validation
    will take place.

    Args:
        fn: The filename of the JSON file to be validated.
        options: An instance of ``ValidationOptions``.

    Returns:
        An instance of FileValidationResults.

    """
    file_results = FileValidationResults(filepath=fn)
    output.info("Performing JSON schema validation on %s" % fn)

    if not options:
        options = ValidationOptions(files=fn)

    try:
        with open(fn) as instance_file:
            file_results.object_results = validate(instance_file, options)

    except Exception as ex:
        if 'Expecting value' in str(ex):
            line_no = str(ex).split()[3]
            file_results.fatal = ValidationErrorResults(
                'Invalid JSON input on line %s' % line_no
            )
        else:
            file_results.fatal = ValidationErrorResults(ex)

        msg = ("Unexpected error occurred with file '{fn}'. No further "
               "validation will be performed: {error}")
        output.info(msg.format(fn=fn, error=str(ex)))

    file_results.is_valid = (all(object_result.is_valid
                                 for object_result in file_results.object_results)
                             and not file_results.fatal)

    return file_results