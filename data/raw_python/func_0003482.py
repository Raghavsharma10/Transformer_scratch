def wait_for(self, timeout):
        """
        A decorator factory that ensures the wrapped function runs in the
        reactor thread.

        When the wrapped function is called, its result is returned or its
        exception raised. Deferreds are handled transparently. Calls will
        timeout after the given number of seconds (a float), raising a
        crochet.TimeoutError, and cancelling the Deferred being waited on.
        """

        def decorator(function):
            @wrapt.decorator
            def wrapper(function, _, args, kwargs):
                @self.run_in_reactor
                def run():
                    return function(*args, **kwargs)

                eventual_result = run()
                try:
                    return eventual_result.wait(timeout)
                except TimeoutError:
                    eventual_result.cancel()
                    raise

            result = wrapper(function)
            # Expose underling function for testing purposes; this attribute is
            # deprecated, use __wrapped__ instead:
            try:
                result.wrapped_function = function
            except AttributeError:
                pass
            return result

        return decorator