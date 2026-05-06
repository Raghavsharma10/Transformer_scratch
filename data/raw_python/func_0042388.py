def once(self):
        """
        Inspected function should be called one time
        Return: self
        """
        def check(): #pylint: disable=missing-docstring
            return super(SinonExpectation, self).calledOnce
        self.valid_list.append(check)
        return self