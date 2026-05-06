def lock_it(lock=g_lock):
    """
    Wrapper. Simple wrapper to make sure a function is only run once at a time.

    .. code: python

        import reusables
        import time

        def func_one(_):
            time.sleep(5)

        @reusables.lock_it()
        def func_two(_):
            time.sleep(5)

        @reusables.time_it(message="test_1 took {0:.2f} seconds")
        def test_1():
            reusables.run_in_pool(func_one, (1, 2, 3), threaded=True)

        @reusables.time_it(message="test_2 took {0:.2f} seconds")
        def test_2():
            reusables.run_in_pool(func_two, (1, 2, 3), threaded=True)

        test_1()
        test_2()

        # test_1 took 5.04 seconds
        # test_2 took 15.07 seconds


    :param lock: Which lock to use, uses unique default
    """
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return func_wrapper