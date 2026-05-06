def remover(self, id_egrupo):
        """Remove um grupo de equipamento a partir do seu identificador.

        :param id_egrupo: Identificador do grupo de equipamento.

        :return: None

        :raise GrupoEquipamentoNaoExisteError: Grupo de equipamento não cadastrado.
        :raise InvalidParameterError: O identificador do grupo é nulo ou inválido.
        :raise GroupDontRemoveError:
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """
        if not is_valid_int_param(id_egrupo):
            raise InvalidParameterError(
                u'O identificador do grupo de equipamento é inválido ou não foi informado.')

        url = 'egrupo/' + str(id_egrupo) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)