def calledWithExactly(cls, spy, *args, **kwargs): #pylint: disable=invalid-name
        """
        Checking the inspector is called with exactly args/kwargs
        Args: SinonSpy, args/kwargs
        """
        cls.__is_spy(spy)
        if not (spy.calledWithExactly(*args, **kwargs)):
            raise cls.failException(cls.message)