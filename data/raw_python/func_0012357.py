def timing_decorator(func):
    """Prints the time func takes to execute."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper for printing execution time.

        Parameters
        ----------
        print_time: bool, optional
            whether or not to print time function takes.
        """
        print_time = kwargs.pop('print_time', False)
        if not print_time:
            return func(*args, **kwargs)
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(func.__name__ + ' took %.3f seconds' %
                  (end_time - start_time))
            return result
    return wrapper