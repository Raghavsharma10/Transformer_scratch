def calledWithMatch(cls, spy, *args, **kwargs): #pylint: disable=invalid-name
        """
        Checking the inspector is called with partial SinonMatcher(args/kwargs)
        Args: SinonSpy, args/kwargs
        """
        cls.__is_spy(spy)
        if not (spy.calledWithMatch(*args, **kwargs)):
            raise cls.failException(cls.message)