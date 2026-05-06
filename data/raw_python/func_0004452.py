def dissociate(self, id_filter, id_eq_type):
        """Removes relationship between Filter and TipoEquipamento.

        :param id_filter: Identifier of Filter. Integer value and greater than zero.
        :param id_eq_type: Identifier of TipoEquipamento. Integer value, greater than zero.

        :return: None

        :raise FilterNotFoundError: Filter not registered.
        :raise TipoEquipamentoNotFoundError: TipoEquipamento not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_filter):
            raise InvalidParameterError(
                u'The identifier of Filter is invalid or was not informed.')

        if not is_valid_int_param(id_eq_type):
            raise InvalidParameterError(
                u'The identifier of TipoEquipamento is invalid or was not informed.')

        url = 'filter/' + \
            str(id_filter) + '/dissociate/' + str(id_eq_type) + '/'

        code, xml = self.submit(None, 'PUT', url)

        return self.response(code, xml)