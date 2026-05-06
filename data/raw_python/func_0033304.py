def trigger(self, name, *args, **kwargs):
        """
        Triggers an event to run through middleware. This method will execute
        a chain of relevant trigger callbacks, until one of the callbacks
        returns the `break_trigger`.
        """

        # Relevant middleware is cached so we don't have to rediscover it
        # every time. Fetch the cached value if possible.

        listeners = self._triggers.get(name, [])

        # Execute each piece of middleware
        for listener in listeners:
            result = listener(*args, **kwargs)

            if result == break_trigger:
                return False

        return True