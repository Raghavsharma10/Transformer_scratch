def evaluate_stop_condition(errdata, stop_condition):
    """
    Call the user-defined function: stop_condition(errdata)
    If the function returns -1, do nothing.  Otherwise, sys.exit.
    """
    if stop_condition:
        return_code = stop_condition(list(errdata))
        if return_code != -1:
            log.info(
                'Stop condition triggered!  Relay is terminating.',
                extra=dict(return_code=return_code))
            sys.exit(return_code)