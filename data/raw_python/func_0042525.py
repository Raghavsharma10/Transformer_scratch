def calledThrice(cls, spy): #pylint: disable=invalid-name
        """
        Checking the inspector is called thrice
        Args: SinonSpy
        """
        cls.__is_spy(spy)
        if not (spy.calledThrice):
            raise cls.failException(cls.message)