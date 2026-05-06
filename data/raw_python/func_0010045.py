def do_until(lambda_expr, timeout=WTF_TIMEOUT_MANAGER.NORMAL, sleep=0.5, message=None):
    '''
    A retry wrapper that'll keep performing the action until it succeeds.
    (main differnce between do_until and wait_until is do_until will keep trying 
    until a value is returned, while wait until will wait until the function 
    evaluates True.)

    Args:
        lambda_expr (lambda) : Expression to evaluate.

    Kwargs: 
        timeout (number): Timeout period in seconds.
        sleep (number) : Sleep time to wait between iterations
        message (str) : Provide a message for TimeoutError raised.

    Returns:
        The value of the evaluated lambda expression.

    Usage::

        do_until(lambda: driver.find_element_by_id("save").click(),
                 timeout=30,
                 sleep=0.5)

    Is equivalent to:

        end_time = datetime.now() + timedelta(seconds=30)
        while datetime.now() < end_time:
            try:
                return driver.find_element_by_id("save").click()
            except:
                pass
            time.sleep(0.5)
        raise OperationTimeoutError()
    '''
    __check_condition_parameter_is_function(lambda_expr)

    end_time = datetime.now() + timedelta(seconds=timeout)
    last_exception = None
    while datetime.now() < end_time:
        try:
            return lambda_expr()
        except Exception as e:
            last_exception = e
            time.sleep(sleep)

    if message:
        raise OperationTimeoutError(message, last_exception)
    else:
        raise OperationTimeoutError("Operation timed out.", last_exception)