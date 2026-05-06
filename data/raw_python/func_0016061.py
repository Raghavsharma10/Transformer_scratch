def print_results_header(identifier, is_valid):
    """Print a header for the results of either a file or an object.

    """
    print_horizontal_rule()
    print_level(logger.info, "[-] Results for: %s", 0, identifier)

    if is_valid:
        marker = _GREEN + "[+]"
        verdict = "Valid"
        log_func = logger.info
    else:
        marker = _RED + "[X]"
        verdict = "Invalid"
        log_func = logger.error
    print_level(log_func, "%s STIX JSON: %s", 0, marker, verdict)