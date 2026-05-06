def on(self, event: str) -> Callable:
        """ Decorator for subscribing a function to a specific event.

        :param event: Name of the event to subscribe to.
        :type event: str

        :return: The outer function.
        :rtype: Callable
        """

        def outer(func):
            self.add_event(func, event)

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return outer