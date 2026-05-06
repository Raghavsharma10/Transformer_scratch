def alterar(
            self,
            id_interface,
            nome,
            protegida,
            descricao,
            id_ligacao_front,
            id_ligacao_back,
            tipo=None,
            vlan=None):
        """Edit an interface by its identifier.

        Equipment identifier is not changed.

        :param nome: Interface name.
        :param protegida: Indication of protected ('0' or '1').
        :param descricao: Interface description.
        :param id_ligacao_front: Front end link interface identifier.
        :param id_ligacao_back: Back end link interface identifier.
        :param id_interface: Interface identifier.

        :return: None

        :raise InvalidParameterError: The parameters interface id, nome and protegida are none or invalid.
        :raise NomeInterfaceDuplicadoParaEquipamentoError: There is already an interface with this name for this equipment.
        :raise InterfaceNaoExisteError: Front link interface and/or back link interface doesn't exist.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        if not is_valid_int_param(id_interface):
            raise InvalidParameterError(
                u'Interface id is invalid or was not informed.')

        url = 'interface/' + str(id_interface) + '/'

        interface_map = dict()
        interface_map['nome'] = nome
        interface_map['protegida'] = protegida
        interface_map['descricao'] = descricao
        interface_map['id_ligacao_front'] = id_ligacao_front
        interface_map['id_ligacao_back'] = id_ligacao_back
        interface_map['tipo'] = tipo
        interface_map['vlan'] = vlan

        code, xml = self.submit({'interface': interface_map}, 'PUT', url)

        return self.response(code, xml)