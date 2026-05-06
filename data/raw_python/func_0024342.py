def intent(self, intent_name):
        """Decorator to register a handler for the given intent.

        The decorated function can either take 0 or 2 arguments. If two are
        specified, it will be provided a dictionary of `{slot_name: value}` and
        a :py:class:`alexandra.session.Session` instance.

        If no session was provided in the request, the session object will be
        `None`. ::

            @alexa_app.intent('FooBarBaz')
            def foo_bar_baz_intent(slots, session):
                pass

            @alexa_app.intent('NoArgs')
            def noargs_intent():
                pass
        """

        # nested decorator so we can have params.
        def _decorator(func):
            arity = func.__code__.co_argcount

            if arity not in [0, 2]:
                raise ValueError("expected 0 or 2 argument function")

            self.intent_map[intent_name] = func
            return func

        return _decorator