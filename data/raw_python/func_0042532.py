def alwaysCalledWithExactly(cls, spy, *args, **kwargs): #pylint: disable=invalid-name
        """
        Checking the inspector is always called with exactly args/kwargs
        Args: SinonSpy, args/kwargs
        """
        cls.__is_spy(spy)
        if not (spy.alwaysCalledWithExactly(*args, **kwargs)):
            raise cls.failException(cls.message)