def execute_with_timeout(
    fn,
    args=None,
    kwargs=None,
    timeout=None,
    fail_if_no_timer=True,
    signal_type=_default_signal_type,
    timer_type=_default_timer_type,
    timeout_exception_cls=TimeoutError,
):
    """
    Executes specified function with timeout. Uses SIGALRM to interrupt it.

    :type fn: function
    :param fn: function to execute

    :type args: tuple
    :param args: function args

    :type kwargs: dict
    :param kwargs: function kwargs

    :type timeout: float
    :param timeout: timeout, seconds; 0 or None means no timeout

    :type fail_if_no_timer: bool
    :param fail_if_no_timer: fail, if timer is nor available; normally it's available only in the
    main thread

    :type signal_type: signalnum
    :param signal_type: type of signal to use (see signal module)

    :type timer_type: signal.ITIMER_REAL, signal.ITIMER_VIRTUAL or signal.ITIMER_PROF
    :param timer_type: type of timer to use (see signal module)

    :type timeout_exception_cls: class
    :param timeout_exception_cls: exception to throw in case of timeout

    :return: fn call result.

    """
    if args is None:
        args = empty_tuple
    if kwargs is None:
        kwargs = empty_dict

    if timeout is None or timeout == 0 or signal_type is None or timer_type is None:
        return fn(*args, **kwargs)

    def signal_handler(signum, frame):
        raise timeout_exception_cls(inspection.get_function_call_str(fn, args, kwargs))

    old_signal_handler = none
    timer_is_set = False
    try:
        try:
            old_signal_handler = signal.signal(signal_type, signal_handler)
            signal.setitimer(timer_type, timeout)
            timer_is_set = True
        except ValueError:
            if fail_if_no_timer:
                raise NotSupportedError(
                    "Timer is not available; the code is probably invoked from outside the main "
                    "thread."
                )
        return fn(*args, **kwargs)
    finally:
        if timer_is_set:
            signal.setitimer(timer_type, 0)
        if old_signal_handler is not none:
            signal.signal(signal_type, old_signal_handler)