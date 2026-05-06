def emit_after(self, event: str) -> Callable:
        """ Decorator that emits events after the function is completed.

        :param event: Name of the event.
        :type event: str

        :return: Callable

        .. note:
            This plainly just calls functions without passing params into the
            subscribed callables. This is great if you want to do some kind
            of post processing without the callable requiring information
            before doing so.
        """

        def outer(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                returned = func(*args, **kwargs)
                self.emit(event)
                return returned

            return wrapper

        return outer