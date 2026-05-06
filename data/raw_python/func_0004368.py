def remover(self, id_equipamento):
        """Remove um equipamento a partir do seu identificador.

        Além de remover o equipamento, a API também remove:

            - O relacionamento do equipamento com os tipos de acessos.
            - O relacionamento do equipamento com os roteiros.
            - O relacionamento do equipamento com os IPs.
            - As interfaces do equipamento.
            - O relacionamento do equipamento com os ambientes.
            - O relacionamento do equipamento com os grupos.

        :param id_equipamento: Identificador do equipamento.

        :return: None

        :raise EquipamentoNaoExisteError: Equipamento não cadastrado.
        :raise InvalidParameterError: O identificador do equipamento é nulo ou inválido.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """
        if not is_valid_int_param(id_equipamento):
            raise InvalidParameterError(
                u'O identificador do equipamento é inválido ou não foi informado.')

        url = 'equipamento/' + str(id_equipamento) + '/'

        code, map = self.submit(None, 'DELETE', url)

        return self.response(code, map)