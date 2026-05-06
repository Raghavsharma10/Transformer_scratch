def twice(self):
        """
        Inspected function should be called two times
        Return: self
        """
        def check(): #pylint: disable=missing-docstring
            return super(SinonExpectation, self).calledTwice
        self.valid_list.append(check)
        return self