def expects(self, prop):
        """
        Adding new property of object as inspector into exp_list
        Args: string (property of object)
        Return: SinonExpectation
        """
        expectation = SinonExpectation(self.obj, prop)
        self.exp_list.append(expectation)
        return expectation