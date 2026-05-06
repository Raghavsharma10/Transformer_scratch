def timeout(seconds=10, default_output='default_output'):
    """ function wrapper that limits the amount of time it has to run
    optional args:
        seconds - how long it has until the function times out
        default_output - what will be returned instead of an error
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            """ throw the custom TimeoutError if called """
            raise TimeoutError(strerror(ETIME))

        def wrapper(*args, **kwargs):
            """ main wrapper for the error """
            # set up the propper error signal
            signal(SIGALRM, _handle_timeout)
            # set the time the function has to run
            alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                if default_output == 'default_output':
                    raise
                else:
                    result = default_output
            finally:
                # cancel the timer
                alarm(0)
            return result

        return wraps(func)(wrapper)
    return decorator