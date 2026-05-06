def handle(event):
        """Decorator for indicating that a given method handles an event.

        Note: while multiple instances of this decorator may be applied to a
        single method, it is not recommended.
        """
        def dec(func):
            if not hasattr(func, '_handle_events'):
                func._handle_events = set()
            func._handle_events.add(event)
            return func

        return dec