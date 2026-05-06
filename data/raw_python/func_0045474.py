def request_generic(self, act, coro, perform):
        """
        Requests an socket operation (in the form of a callable `perform` that
        does the actual socket system call) for `coro` corutine with parameters
        and completion passed via `act`.

        The socket operation request parameters are passed in `act`.
        When request is completed the results will be set in `act`.

        Note: `act` is usualy a SocketOperation instance and the request_foo
        calls are usually made from a Foo subclass.
        """
        result = self.multiplex_first and self.try_run_act(act, perform)
        if result:
            return result, coro
        else:
            self.add_token(act, coro, perform)