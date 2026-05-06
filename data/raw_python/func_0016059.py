def print_schema_results(results, level=0):
    """Print JSON Schema validation errors to stdout.

    Args:
        results: An instance of ObjectValidationResults.
        level: The level at which to print the results.

    """
    for error in results.errors:
        print_level(logger.error, _RED + "[X] %s", level, error)