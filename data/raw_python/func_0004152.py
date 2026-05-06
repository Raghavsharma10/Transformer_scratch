def alterar(self, id_divisiondc, name):
        """Change Division Dc from by the identifier.

        :param id_divisiondc: Identifier of the Division Dc. Integer value and greater than zero.
        :param name: Division Dc name. String with a minimum 2 and maximum of 80 characters

        :return: None

        :raise InvalidParameterError: The identifier of Division Dc or name is null and invalid.
        :raise NomeDivisaoDcDuplicadoError: There is already a registered Division Dc with the value of name.
        :raise DivisaoDcNaoExisteError: Division Dc not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_divisiondc):
            raise InvalidParameterError(
                u'The identifier of Division Dc is invalid or was not informed.')

        url = 'divisiondc/' + str(id_divisiondc) + '/'

        division_dc_map = dict()
        division_dc_map['name'] = name

        code, xml = self.submit({'division_dc': division_dc_map}, 'PUT', url)

        return self.response(code, xml)