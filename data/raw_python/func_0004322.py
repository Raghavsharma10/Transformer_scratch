def inserir(
            self,
            nome,
            protegida,
            descricao,
            id_ligacao_front,
            id_ligacao_back,
            id_equipamento,
            tipo=None,
            vlan=None):
        """Insert new interface for an equipment.

        :param nome: Interface name.
        :param protegida: Indication of protected ('0' or '1').
        :param descricao: Interface description.
        :param id_ligacao_front: Front end link interface identifier.
        :param id_ligacao_back: Back end link interface identifier.
        :param id_equipamento: Equipment identifier.

        :return: Dictionary with the following: {'interface': {'id': < id >}}

        :raise EquipamentoNaoExisteError: Equipment does not exist.
        :raise InvalidParameterError: The parameters nome, protegida and/or equipment id are none or invalid.
        :raise NomeInterfaceDuplicadoParaEquipamentoError: There is already an interface with this name for this equipment.
        :raise InterfaceNaoExisteError: Front link interface and/or back link interface doesn't exist.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        interface_map = dict()
        interface_map['nome'] = nome
        interface_map['protegida'] = protegida
        interface_map['descricao'] = descricao
        interface_map['id_ligacao_front'] = id_ligacao_front
        interface_map['id_ligacao_back'] = id_ligacao_back
        interface_map['id_equipamento'] = id_equipamento
        interface_map['tipo'] = tipo
        interface_map['vlan'] = vlan

        code, xml = self.submit(
            {'interface': interface_map}, 'POST', 'interface/')
        return self.response(code, xml)