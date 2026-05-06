def _print_exception_list(exceptions, message, output_file=None):
    """
    Given a list of sys.exc_info tuples, print them all using the traceback
    module preceeded by a message and separated by a blank line.
    """
    output_file = output_file or sys.stderr
    message = 'Exception %s:\n' % message
    for exc_type, exc_value, exc_traceback in exceptions:
        output_file.write(message)
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  file=output_file)
        output_file.write('\n')