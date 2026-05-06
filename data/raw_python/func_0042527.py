def callOrder(cls, *args): #pylint: disable=invalid-name
        """
        Checking the inspector is called with given priority
        Args: SinonSpy, list of inspectors
        eg.
            [spy1, spy2, spy3] => spy1 is called before spy2, spy2 is called before spy3
            [spy1, spy2, spy1] => spy1 is called before and after spy2
        """
        for spy in args:
            cls.__is_spy(spy)
        for idx, val in enumerate(args):
            if val != args[0]:
                if not (val.calledAfter(args[idx-1])):
                    raise cls.failException(cls.message)
            if val != args[-1]:
                if not (val.calledBefore(args[idx+1])):
                    raise cls.failException(cls.message)