def thrice(self):
        """
        Inspected function should be called three times
        Return: self
        """

        def check(): #pylint: disable=missing-docstring
            return super(SinonExpectation, self).calledThrice
        self.valid_list.append(check)
        return self