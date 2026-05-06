def remover(self, id_logicalenvironment):
        """Remove Logical Environment from by the identifier.

        :param id_logicalenvironment: Identifier of the Logical Environment. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Logical Environment is null and invalid.
        :raise AmbienteLogicoNaoExisteError: Logical Environment not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_logicalenvironment):
            raise InvalidParameterError(
                u'The identifier of Logical Environment is invalid or was not informed.')

        url = 'logicalenvironment/' + str(id_logicalenvironment) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)