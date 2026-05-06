def report(title='Unhandled Exception', exec_info=(), **kwargs):
    """
    Create a technical server error response. The last three arguments are
    the values returned from sys.exc_info() and friends.

    :param title: Title of error email
    :type title: str
    :param exec_info: exc_info from traceback
    """

    exc_type, exc_value, tb = exec_info or sys.exc_info()
    reporter = ExceptionReporter(exc_type, exc_value, tb)
    html = reporter.get_traceback_html(**kwargs)

    mail_admins(title, 'html only', html_message=html)