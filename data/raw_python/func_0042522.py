def notCalled(cls, spy): #pylint: disable=invalid-name
        """
        Checking the inspector is not called
        Args: SinonSpy
        """
        cls.__is_spy(spy)
        if not (not spy.called):
            raise cls.failException(cls.message)