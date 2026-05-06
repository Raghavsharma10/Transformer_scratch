def alterar(self, id_logicalenvironment, name):
        """Change Logical Environment from by the identifier.

        :param id_logicalenvironment: Identifier of the Logical Environment. Integer value and greater than zero.
        :param name: Logical Environment name. String with a minimum 2 and maximum of 80 characters

        :return: None

        :raise InvalidParameterError: The identifier of Logical Environment or name is null and invalid.
        :raise NomeAmbienteLogicoDuplicadoError: There is already a registered Logical Environment with the value of name.
        :raise AmbienteLogicoNaoExisteError: Logical Environment not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_logicalenvironment):
            raise InvalidParameterError(
                u'The identifier of Logical Environment is invalid or was not informed.')

        url = 'logicalenvironment/' + str(id_logicalenvironment) + '/'

        logical_environment_map = dict()
        logical_environment_map['name'] = name

        code, xml = self.submit(
            {'logical_environment': logical_environment_map}, 'PUT', url)

        return self.response(code, xml)