def get_unresolved_variables(f):
    """
    Gets unresolved vars from file
    """
    reporter = RReporter()
    checkPath(f, reporter=reporter)
    return dict(reporter.messages)