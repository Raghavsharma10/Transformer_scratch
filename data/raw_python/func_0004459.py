def inserir(self, name):
        """Inserts a new Logical Environment and returns its identifier.

        :param name: Logical Environment name. String with a minimum 2 and maximum of 80 characters

        :return: Dictionary with the following structure:

        ::

            {'logical_environment': {'id': < id_logical_environment >}}

        :raise InvalidParameterError: Name is null and invalid.
        :raise NomeAmbienteLogicoDuplicadoError: There is already a registered Logical Environment with the value of name.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        logical_environment_map = dict()
        logical_environment_map['name'] = name

        code, xml = self.submit(
            {'logical_environment': logical_environment_map}, 'POST', 'logicalenvironment/')

        return self.response(code, xml)