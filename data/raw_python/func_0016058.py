def print_fatal_results(results, level=0):
    """Print fatal errors that occurred during validation runs.
    """
    print_level(logger.critical, _RED + "[X] Fatal Error: %s", level, results.error)