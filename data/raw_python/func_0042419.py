def returns(self, obj):
        """
        Customizes the return values of the stub function. If conditions like withArgs or onCall
        were specified, then the return value will only be returned when the conditions are met.

        Args: obj (anything)
        Return: a SinonStub object (able to be chained)
        """
        self._copy._append_condition(self, lambda *args, **kwargs: obj)
        return self