def yield_event(self, act):
        """
        Hande completion for a request and return an (op, coro) to be
        passed to the scheduler on the last completion loop of a proactor.
        """
        if act in self.tokens:
            coro = act.coro
            op = self.try_run_act(act, self.tokens[act])
            if op:
                del self.tokens[act]
                return op, coro