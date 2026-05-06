def remove(self, id_equipamento, id_egrupo):
        """Remove a associacao de um grupo de equipamento com um equipamento a partir do seu identificador.

        :param id_egrupo: Identificador do grupo de equipamento.
        :param id_equipamento: Identificador do equipamento.

        :return: None

        :raise EquipamentoNaoExisteError: Equipamento não cadastrado.
        :raise GrupoEquipamentoNaoExisteError: Grupo de equipamento não cadastrado.
        :raise InvalidParameterError: O identificador do grupo é nulo ou inválido.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if not is_valid_int_param(id_egrupo):
            raise InvalidParameterError(
                u'O identificador do grupo de equipamento é inválido ou não foi informado.')

        if not is_valid_int_param(id_equipamento):
            raise InvalidParameterError(
                u'O identificador do equipamento é inválido ou não foi informado.')

        url = 'egrupo/equipamento/' + \
            str(id_equipamento) + '/egrupo/' + str(id_egrupo) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)