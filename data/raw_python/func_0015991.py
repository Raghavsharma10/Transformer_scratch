def run_validation(options):
    """Validate files based on command line options.

    Args:
        options: An instance of ``ValidationOptions`` containing options for
            this validation run.

    """
    if options.files == sys.stdin:
        results = validate(options.files, options)
        return [FileValidationResults(is_valid=results.is_valid,
                                      filepath='stdin',
                                      object_results=results)]

    files = get_json_files(options.files, options.recursive)

    results = [validate_file(fn, options) for fn in files]

    return results