def _call_atfork_list(call_list):
    """
    Given a list of callables in call_list, call them all in order and save
    and return a list of sys.exc_info() tuples for each exception raised.
    """
    exception_list = []
    for func in call_list:
        try:
            func()
        except:
            exception_list.append(sys.exc_info())
    return exception_list