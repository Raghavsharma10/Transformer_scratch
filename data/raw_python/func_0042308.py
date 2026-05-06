def calledBefore(self, spy): #pylint: disable=invalid-name
        """
        Compares the order in which two spies were called

        E.g.
            spy_a()
            spy_b()
            spy_a.calledBefore(spy_b) # True
            spy_b.calledBefore(spy_a) # False
            spy_a()
            spy_b.calledBefore(spy_a) # True

        Args: a Spy to compare with
        Return: Boolean True if this spy's first call was called before the given spy's last call
        """
        this_call = self.firstCall if self.firstCall is not None else False
        given_call = spy.lastCall if spy.lastCall is not None else False
        return (this_call and not given_call) or (this_call and given_call and this_call.callId < given_call.callId)