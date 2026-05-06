def _gather_exposed_methods(self):
        """
        Searches for the exposed methods in the current microservice class. A method is considered
        exposed if it is decorated with the :py:func:`gemstone.public_method` or
        :py:func:`gemstone.private_api_method`.
        """

        self._extract_methods_from_container(self)

        for module in self.modules:
            self._extract_methods_from_container(module)