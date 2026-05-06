def queue_it(queue=g_queue, **put_args):
    """
    Wrapper. Instead of returning the result of the function, add it to a queue.

    .. code: python

        import reusables
        import queue

        my_queue = queue.Queue()

        @reusables.queue_it(my_queue)
        def func(a):
            return a

        func(10)

        print(my_queue.get())
        # 10


    :param queue: Queue to add result into
    """
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            queue.put(func(*args, **kwargs), **put_args)
        return wrapper
    return func_wrapper