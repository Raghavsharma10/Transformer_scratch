def list_connections(self, nome_interface, id_equipamento):
        """List interfaces linked to back and front of specified interface.

        :param nome_interface: Interface name.
        :param id_equipamento: Equipment identifier.

        :return: Dictionary with the following:

        ::

            {'interfaces':[ {'id': < id >,
            'interface': < nome >,
            'descricao': < descricao >,
            'protegida': < protegida >,
            'equipamento': < id_equipamento >,
            'tipo_equip': < id_tipo_equipamento >,
            'ligacao_front': < id_ligacao_front >,
            'nome_ligacao_front': < interface_name >,
            'nome_equip_l_front': < equipment_name >,
            'ligacao_back': < id_ligacao_back >,
            'nome_ligacao_back': < interface_name >,
            'nome_equip_l_back': < equipment_name > }, ... other interfaces ...]}

        :raise InterfaceNaoExisteError: Interface doesn't exist or is not associated with this equipment.
        :raise EquipamentoNaoExisteError: Equipment doesn't exist.
        :raise InvalidParameterError: Interface name and/or equipment identifier are none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_equipamento):
            raise InvalidParameterError(
                u'Equipment identifier is none or was not informed.')

        if (nome_interface is None) or (nome_interface == ''):
            raise InvalidParameterError(u'Interface name was not informed.')

        # Temporário, remover. Fazer de outra forma.
        nome_interface = nome_interface.replace('/', 's2it_replace')

        url = 'interface/' + \
            urllib.quote(nome_interface) + '/equipment/' + \
            str(id_equipamento) + '/'

        code, map = self.submit(None, 'GET', url)

        key = 'interfaces'
        return get_list_map(self.response(code, map, [key]), key)