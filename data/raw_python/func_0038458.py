def map(self, fn, *iterables, **kwargs):
        """Returns a iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        loop = evergreen.current.loop

        timeout = kwargs.get('timeout')
        if timeout is not None:
            end_time = timeout + loop.time()

        fs = [self.submit(fn, *args) for args in zip(*iterables)]

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                for future in fs:
                    if timeout is None:
                        yield future.get()
                    else:
                        yield future.get(end_time - loop.time())
            finally:
                for future in fs:
                    future.cancel()
        return result_iterator()