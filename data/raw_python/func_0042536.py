def threw(cls, spy, error_type=None):
        """
        Checking the inspector is raised error_type
        Args: SinonSpy, Exception (defaut: None)
        """
        cls.__is_spy(spy)
        if not (spy.threw(error_type)):
            raise cls.failException(cls.message)