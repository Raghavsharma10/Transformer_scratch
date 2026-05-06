def calledWith(cls, spy, *args, **kwargs): #pylint: disable=invalid-name
        """
        Checking the inspector is called with partial args/kwargs
        Args: SinonSpy, args/kwargs
        """
        cls.__is_spy(spy)
        if not (spy.calledWith(*args, **kwargs)):
            raise cls.failException(cls.message)