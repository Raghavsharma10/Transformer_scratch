def execute(self, method, args, kwargs):
        """
        Execute the given method and stores its result.
        The result is considered "done" even if the method raises an exception

        :param method: The method to execute
        :param args: Method positional arguments
        :param kwargs: Method keyword arguments
        :raise Exception: The exception raised by the method
        """
        # Normalize arguments
        if args is None:
            args = []

        if kwargs is None:
            kwargs = {}

        try:
            # Call the method
            self._result = method(*args, **kwargs)

        except Exception as ex:
            self._exception = ex
            raise

        finally:
            # Mark the action as executed
            self._done_event.set()