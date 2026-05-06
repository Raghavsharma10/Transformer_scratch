def calledTwice(cls, spy): #pylint: disable=invalid-name
        """
        Checking the inspector is called twice
        Args: SinonSpy
        """
        cls.__is_spy(spy)
        if not (spy.calledTwice):
            raise cls.failException(cls.message)