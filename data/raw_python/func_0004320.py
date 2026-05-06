def listar_por_equipamento(self, id_equipamento):
        """List all interfaces of an equipment.

        :param id_equipamento: Equipment identifier.

        :return: Dictionary with the following:

        ::

            {'interface':
            [{'protegida': < protegida >,
            'nome': < nome >,
            'id_ligacao_front': < id_ligacao_front >,
            'id_equipamento': < id_equipamento >,
            'id': < id >,
            'descricao': < descricao >,
            'id_ligacao_back': < id_ligacao_back >}, ... other interfaces ...]}

        :raise InvalidParameterError: Equipment identifier is invalid or none.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_equipamento):
            raise InvalidParameterError(
                u'Equipment id is invalid or was not informed.')

        url = 'interface/equipamento/' + str(id_equipamento) + '/'

        code, map = self.submit(None, 'GET', url)

        key = 'interface'
        return get_list_map(self.response(code, map, [key]), key)