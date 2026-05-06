def alwaysThrew(cls, spy, error_type=None): #pylint: disable=invalid-name
        """
        Checking the inspector is always raised error_type
        Args: SinonSpy, Exception (defaut: None)
        """
        cls.__is_spy(spy)
        if not (spy.alwaysThrew(error_type)):
            raise cls.failException(cls.message)