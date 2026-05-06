def remover(self, id_tipo_acesso, id_equipamento):
        """Removes relationship between equipment and access type.

        :param id_equipamento: Equipment identifier.
        :param id_tipo_acesso: Access type identifier.

        :return: None

        :raise EquipamentoNaoExisteError: Equipment doesn't exist.
        :raise EquipamentoAcessoNaoExisteError: Relationship between equipment and access type doesn't exist.
        :raise InvalidParameterError: Equipment and/or access type id is/are invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_tipo_acesso):
            raise InvalidParameterError(u'Access type id is invalid.')

        if not is_valid_int_param(id_equipamento):
            raise InvalidParameterError(u'Equipment id is invalid.')

        url = 'equipamentoacesso/' + \
            str(id_equipamento) + '/' + str(id_tipo_acesso) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)