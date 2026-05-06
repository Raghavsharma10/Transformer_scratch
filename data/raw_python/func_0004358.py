def criar_ip(self, id_vlan, id_equipamento, descricao):
        """Aloca um IP em uma VLAN para um equipamento.

        Insere um novo IP para a VLAN e o associa ao equipamento.

        :param id_vlan: Identificador da vlan.
        :param id_equipamento: Identificador do equipamento.
        :param descricao: Descriçao do IP.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'ip': {'id': < id_ip >,
            'id_network_ipv4': < id_network_ipv4 >,
            'oct1’: < oct1 >,
            'oct2': < oct2 >,
            'oct3': < oct3 >,
            'oct4': < oct4 >,
            'descricao': < descricao >}}

        :raise InvalidParameterError: O identificador da VLAN e/ou do equipamento são nulos ou inválidos.
        :raise EquipamentoNaoExisteError: Equipamento não cadastrado.
        :raise VlanNaoExisteError: VLAN não cadastrada.
        :raise IPNaoDisponivelError: Não existe IP disponível para a VLAN informada.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """
        ip_map = dict()
        ip_map['id_vlan'] = id_vlan
        ip_map['descricao'] = descricao
        ip_map['id_equipamento'] = id_equipamento

        code, xml = self.submit({'ip': ip_map}, 'POST', 'ip/')

        return self.response(code, xml)