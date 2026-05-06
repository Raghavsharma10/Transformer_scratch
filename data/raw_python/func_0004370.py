def remover_grupo(self, id_equipamento, id_grupo):
        """Remove a associação de um equipamento com um grupo de equipamento.

        :param id_equipamento: Identificador do equipamento.
        :param id_grupo: Identificador do grupo de equipamento.

        :return: None

        :raise EquipamentoGrupoNaoExisteError: Associação entre grupo e equipamento não cadastrada.
        :raise EquipamentoNaoExisteError: Equipamento não cadastrado.
        :raise EquipmentDontRemoveError: Failure to remove an association between an equipment and a group because the group is related only to a group.
        :raise InvalidParameterError: O identificador do equipamento e/ou do grupo são nulos ou inválidos.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if not is_valid_int_param(id_equipamento):
            raise InvalidParameterError(
                u'O identificador do equipamento é inválido ou não foi informado.')

        if not is_valid_int_param(id_grupo):
            raise InvalidParameterError(
                u'O identificador do grupo é inválido ou não foi informado.')

        url = 'equipamentogrupo/equipamento/' + \
            str(id_equipamento) + '/egrupo/' + str(id_grupo) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)