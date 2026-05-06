def wait_until(condition, timeout=WTF_TIMEOUT_MANAGER.NORMAL, sleep=0.5, pass_exceptions=False, message=None):
    '''
    Waits wrapper that'll wait for the condition to become true.
    (main differnce between do_until and wait_until is do_until will keep trying 
    until a value is returned, while wait until will wait until the function 
    evaluates True.)

    Args:
        condition (lambda) - Lambda expression to wait for to evaluate to True.

    Kwargs:
        timeout (number) : Maximum number of seconds to wait.
        sleep (number) : Sleep time to wait between iterations.
        pass_exceptions (bool) : If set true, any exceptions raised will be re-raised up the chain.
                                Normally exceptions are ignored.
        message (str) : Optional message to pass into OperationTimeoutError if the wait times out.

    Example::

        wait_until(lambda: driver.find_element_by_id("success").is_displayed(), 
                   timeout=30,
                   sleep=0.5)

    is equivalent to::

        end_time = datetime.now() + timedelta(seconds=30)
        did_succeed = False
        while datetime.now() < end_time:
            try:
                if driver.find_element_by_id("success").is_displayed():
                    did_succeed = True
                    break;
            except:
                pass
            time.sleep(0.5)
        if not did_succeed:
            raise OperationTimeoutError()
    '''
    __check_condition_parameter_is_function(condition)

    last_exception = None
    end_time = datetime.now() + timedelta(seconds=timeout)
    while datetime.now() < end_time:
        try:
            if condition():
                return
        except Exception as e:
            if pass_exceptions:
                raise e
            else:
                last_exception = e
        time.sleep(sleep)

    if message:
        if last_exception:
            raise OperationTimeoutError(message, e)
        else:
            raise OperationTimeoutError(message)
    else:
        if last_exception:
            raise OperationTimeoutError("Operation timed out.", e)
        else:
            raise OperationTimeoutError("Operation timed out.")