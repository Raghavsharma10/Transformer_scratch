def calledWith(self, *args, **kwargs): #pylint: disable=invalid-name
        """
        Determining whether args/kwargs are called previously
        Eg.
            f(1, 2, 3)
            spy.calledWith(1, 2) will return True, because they are called partially
            f(a=1, b=2, c=3)
            spy.calledWith(a=1, b=3) will return True, because they are called partially
        Return: Boolean
        """
        self.__get_func = SinonSpy.__get_directly
        return self.calledWithMatch(*args, **kwargs)