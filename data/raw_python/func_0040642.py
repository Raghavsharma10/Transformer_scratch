def trace(fn=None, profiler=None) -> Callable:
    ''' This decorator allows you to visually trace
        the steps of a function as it executes to see
        what happens to the data as things are being
        processed.

        If you want to use a custom profiler, use the
        @trace(profiler=my_custom_profiler) syntax.

        Example Usage:

            def count_to(target):
               for i in range(1, target+1):
                   yield i

            @trace
            def sum_of_count(target):
               total = 0
               for i in count_to(target):
                   total += i
               return total

            sum_of_count(10)
    '''
    # analyze usage
    custom_profiler = fn is None and profiler is not None
    no_profiler = profiler is None and fn is not None
    no_args = profiler is None and fn is None
    # adjust for usage
    if custom_profiler: # for @trace(profiler=...)
        return partial(trace, profiler=profiler)
    elif no_args: # for @trace()
        return trace
    elif no_profiler: # for @trace
        profiler = default_profiler
    # validate input
    assert callable(fn)
    assert callable(profiler)
    # build the decorator
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # flag for default_profiler to know to ignore this scope
        wafflestwaffles = None
        # save the previous profiler
        old_profiler = sys.getprofile()
        # set the new profiler
        sys.setprofile(profiler)
        try:
            # run the function
            return fn(*args, **kwargs)
        finally:
            # revert the profiler
            sys.setprofile(old_profiler)
    return wrapper