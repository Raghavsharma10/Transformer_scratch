def associate(self, et_id, id_filter):
        """Create a relationship between Filter and TipoEquipamento.

        :param et_id: Identifier of TipoEquipamento. Integer value and greater than zero.
        :param id_filter: Identifier of Filter. Integer value and greater than zero.

        :return: Following dictionary:

        ::

            {'equiptype_filter_xref': {'id': < id_equiptype_filter_xref >} }

        :raise InvalidParameterError: TipoEquipamento/Filter identifier is null and/or invalid.
        :raise TipoEquipamentoNaoExisteError: TipoEquipamento not registered.
        :raise FilterNotFoundError: Filter not registered.
        :raise FilterEqTypeAssociationError: TipoEquipamento and Filter already associated.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(et_id):
            raise InvalidParameterError(
                u'The identifier of TipoEquipamento is invalid or was not informed.')

        if not is_valid_int_param(id_filter):
            raise InvalidParameterError(
                u'The identifier of Filter is invalid or was not informed.')

        url = 'filter/' + str(id_filter) + '/equiptype/' + str(et_id) + '/'

        code, xml = self.submit(None, 'PUT', url)

        return self.response(code, xml)