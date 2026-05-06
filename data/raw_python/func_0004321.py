def get_by_id(self, id_interface):
        """Get an interface by id.

        :param id_interface: Interface identifier.

        :return: Following dictionary:

        ::

            {'interface': {'id': < id >,
            'interface': < interface >,
            'descricao': < descricao >,
            'protegida': < protegida >,
            'tipo_equip': < id_tipo_equipamento >,
            'equipamento': < id_equipamento >,
            'equipamento_nome': < nome_equipamento >
            'ligacao_front': < id_ligacao_front >,
            'nome_ligacao_front': < interface_name >,
            'nome_equip_l_front': < equipment_name >,
            'ligacao_back': < id_ligacao_back >,
            'nome_ligacao_back': < interface_name >,
            'nome_equip_l_back': < equipment_name > }}

        :raise InvalidParameterError: Interface identifier is invalid or none.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_interface):
            raise InvalidParameterError(
                u'Interface id is invalid or was not informed.')

        url = 'interface/' + str(id_interface) + '/get/'

        code, map = self.submit(None, 'GET', url)

        return self.response(code, map)