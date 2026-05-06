def result(self):
        """The result method which returns a context manager

        Returns:
            ContextManager: The corresponding context manager
        """
        if self.exception():
            raise self.exception()
        # Otherwise return a context manager that cleans up after the block.

        @contextlib.contextmanager
        def f():
            try:
                yield self._wrapped.result()
            finally:
                self._exit_callback()
        return f()