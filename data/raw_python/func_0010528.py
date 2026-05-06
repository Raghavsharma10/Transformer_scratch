def task(self, queue: str = 'kuyruk', **kwargs: Any) -> Callable:
        """
        Wrap functions with this decorator to convert them to *tasks*.
        After wrapping, calling the function will send a message to
        a queue instead of running the function.

        :param queue: Queue name for the tasks.
        :param kwargs: Keyword arguments will be passed to
            :class:`~kuyruk.Task` constructor.
        :return: Callable :class:`~kuyruk.Task` object wrapping the original
            function.

        """
        def wrapper(f: Callable) -> Task:
            return Task(f, self, queue, **kwargs)

        return wrapper