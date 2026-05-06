def associa_equipamento(self, id_equip, id_grupo_equipamento):
        """Associa um equipamento a um grupo.

        :param id_equip: Identificador do equipamento.
        :param id_grupo_equipamento: Identificador do grupo de equipamento.

        :return: Dicionário com a seguinte estrutura:
            {'equipamento_grupo':{'id': < id_equip_do_grupo >}}

        :raise GrupoEquipamentoNaoExisteError: Grupo de equipamento não cadastrado.
        :raise InvalidParameterError: O identificador do equipamento e/ou do grupo são nulos ou inválidos.
        :raise EquipamentoNaoExisteError: Equipamento não cadastrado.
        :raise EquipamentoError: Equipamento já está associado ao grupo.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        equip_map = dict()
        equip_map['id_grupo'] = id_grupo_equipamento
        equip_map['id_equipamento'] = id_equip

        code, xml = self.submit(
            {'equipamento_grupo': equip_map}, 'POST', 'equipamentogrupo/associa/')

        return self.response(code, xml)