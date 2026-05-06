def remover(self, id_divisiondc):
        """Remove Division Dc from by the identifier.

        :param id_divisiondc: Identifier of the Division Dc. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: The identifier of Division Dc is null and invalid.
        :raise DivisaoDcNaoExisteError: Division Dc not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_divisiondc):
            raise InvalidParameterError(
                u'The identifier of Division Dc is invalid or was not informed.')

        url = 'divisiondc/' + str(id_divisiondc) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)