def withArgs(self, *args, **kwargs): #pylint: disable=invalid-name
        """
        Inspected function should be called with some of specified arguments
        Args: any
        Return: self
        """
        def check(): #pylint: disable=missing-docstring
            return super(SinonExpectation, self).calledWith(*args, **kwargs)
        self.valid_list.append(check)
        return self