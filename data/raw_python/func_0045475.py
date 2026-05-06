def add_token(self, act, coro, performer):
        """
        Adds a completion token `act` in the proactor with associated `coro`
        corutine and perform callable.
        """
        assert act not in self.tokens
        act.coro = coro
        self.tokens[act] = performer
        self.register_fd(act, performer)