def print_warning_results(results, level=0):
    """Print warning messages found during validation.
    """
    marker = _YELLOW + "[!] "

    for warning in results.warnings:
        print_level(logger.warning, marker + "Warning: %s", level, warning)