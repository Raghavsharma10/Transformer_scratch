def print_results(results):
    """Print `results` (the results of validation) to stdout.

    Args:
        results: A list of FileValidationResults or ObjectValidationResults
                 instances.

    """
    if not isinstance(results, list):
        results = [results]

    for r in results:
        try:
            r.log()
        except AttributeError:
            raise ValueError('Argument to print_results() must be a list of '
                             'FileValidationResults or ObjectValidationResults.')