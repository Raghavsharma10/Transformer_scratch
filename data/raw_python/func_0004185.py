def add_equipamento(
            self,
            id_tipo_equipamento,
            id_modelo,
            prefixo,
            id_grupo,
            id_vlan,
            descricao_vlan):
        """Adiciona um equipamento na lista de equipamentos para operação de inserir/alterar um grupo virtual.

        :param id_tipo_equipamento: Identificador do tipo de equipamento.
        :param id_modelo: Identificador do modelo do equipamento.
        :param prefixo: Prefixo do nome do equipamento.
        :param id_grupo: Identificador do grupo do equipamento.
        :param id_vlan: Identificador da VLAN para criar um IP para o equipamento.
        :param descricao_vlan: Descrição do IP que será criado.

        :return: None
        """

        equipamento_map = dict()

        equipamento_map['id_tipo_equipamento'] = id_tipo_equipamento
        equipamento_map['id_modelo'] = id_modelo
        equipamento_map['prefixo'] = prefixo
        equipamento_map['id_grupo'] = id_grupo
        equipamento_map['ip'] = {
            'id_vlan': id_vlan,
            'descricao': descricao_vlan}

        self.lista_equipamentos.append(equipamento_map)