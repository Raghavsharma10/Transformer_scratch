def get_exception():
    """Return full formatted traceback as a string."""
    trace = ""
    exception = ""
    exc_list = traceback.format_exception_only(
        sys.exc_info()[0], sys.exc_info()[1]
    )
    for entry in exc_list:
        exception += entry
    tb_list = traceback.format_tb(sys.exc_info()[2])
    for entry in tb_list:
        trace += entry
    return "%s\n%s" % (exception, trace)