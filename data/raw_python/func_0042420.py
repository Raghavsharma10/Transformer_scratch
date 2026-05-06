def throws(self, exception=Exception):
        """
        Customizes the stub function to raise an exception. If conditions like withArgs or onCall
        were specified, then the return value will only be returned when the conditions are met.

        Args: exception (by default=Exception, it could be any customized exception)
        Return: a SinonStub object (able to be chained)
        """
        def exception_function(*args, **kwargs):
            raise exception
        self._copy._append_condition(self, exception_function)
        return self