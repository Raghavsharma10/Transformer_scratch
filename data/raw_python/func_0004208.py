def remover(self, id_direito):
        """Remove os direitos de um grupo de usuário em um grupo de equipamento a partir do seu identificador.

        :param id_direito: Identificador do direito grupo equipamento

        :return: None

        :raise DireitoGrupoEquipamentoNaoExisteError: Direito Grupo Equipamento não cadastrado.
        :raise InvalidParameterError: O identificador do direito grupo equipamento é nulo ou inválido.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """
        if not is_valid_int_param(id_direito):
            raise InvalidParameterError(
                u'O identificador do direito grupo equipamento é inválido ou não foi informado.')

        url = 'direitosgrupoequipamento/' + str(id_direito) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)