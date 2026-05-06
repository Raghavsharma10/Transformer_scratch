def callCount(cls, spy, number): #pylint: disable=invalid-name
        """
        Checking the inspector is called number times
        Args: SinonSpy, number
        """
        cls.__is_spy(spy)
        if not (spy.callCount == number):
            raise cls.failException(cls.message)