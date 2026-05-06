def inserir(self, type, description):
        """Inserts a new Script Type and returns its identifier.

        :param type: Script Type type. String with a minimum 3 and maximum of 40 characters
        :param description: Script Type description. String with a minimum 3 and maximum of 100 characters

        :return: Dictionary with the following structure:

        ::

            {'script_type': {'id': < id_script_type >}}

        :raise InvalidParameterError: Type or description is null and invalid.
        :raise NomeTipoRoteiroDuplicadoError: Type script already registered with informed.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        script_type_map = dict()
        script_type_map['type'] = type
        script_type_map['description'] = description

        code, xml = self.submit(
            {'script_type': script_type_map}, 'POST', 'scripttype/')

        return self.response(code, xml)