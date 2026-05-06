def alwaysCalledWith(self, *args, **kwargs): #pylint: disable=invalid-name
        """
        Determining whether args/kwargs are the ONLY args/kwargs called previously
        Eg.
            f(1, 2, 3)
            f(1, 2, 3)
            spy.alwaysCalledWith(1, 2) will return True, because they are the ONLY called args
            f(1, 3)
            spy.alwaysCalledWith(1) will return True, because 1 is the ONLY called args
            spy.alwaysCalledWith(1, 2) will return False, because 2 is not the ONLY called args
        Return: Boolean
        """
        self.__get_func = SinonSpy.__get_directly
        return self.alwaysCalledWithMatch(*args, **kwargs)