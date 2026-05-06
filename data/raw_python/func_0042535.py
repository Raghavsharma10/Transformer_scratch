def neverCalledWithMatch(cls, spy, *args, **kwargs): #pylint: disable=invalid-name
        """
        Checking the inspector is never called with partial SinonMatcher(args/kwargs)
        Args: SinonSpy, args/kwargs
        """
        cls.__is_spy(spy)
        if not (spy.neverCalledWithMatch(*args, **kwargs)):
            raise cls.failException(cls.message)