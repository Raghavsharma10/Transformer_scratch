def exactly(self, number):
        """
        Inspected function should be called exactly number times
        Return: self
        """
        def check(): #pylint: disable=missing-docstring
            return True if number == super(SinonExpectation, self).callCount else False
        self.valid_list.append(check)
        return self