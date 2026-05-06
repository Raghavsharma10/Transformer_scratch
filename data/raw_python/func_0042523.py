def calledOnce(cls, spy): #pylint: disable=invalid-name
        """
        Checking the inspector is called once
        Args: SinonSpy
        """
        cls.__is_spy(spy)
        if not (spy.calledOnce):
            raise cls.failException(cls.message)