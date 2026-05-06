def never(self):
        """
        Inspected function should never be called
        Return: self
        """
        def check(): #pylint: disable=missing-docstring
            return not super(SinonExpectation, self).called
        self.valid_list.append(check)
        return self