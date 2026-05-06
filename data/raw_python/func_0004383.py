def listar_healthcheck_expect(self, id_ambiente):
        """Lista os healthcheck_expect´s de um ambiente.

        :param id_ambiente: Identificador do ambiente.

        :return: Dicionário com a seguinte estrutura:

        ::

            {'healthcheck_expect': [{'id': < id_healthcheck_expect >,
             'expect_string': < expect_string >,
             'match_list': < match_list >,
             'id_ambiente': < id_ambiente >},
             ... demais healthcheck_expects ...]}

        :raise InvalidParameterError: O identificador do ambiente é nulo ou inválido.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao gerar o XML de resposta.
        """

        if not is_valid_int_param(id_ambiente):
            raise InvalidParameterError(
                u'O identificador do ambiente é inválido ou não foi informado.')

        url = 'healthcheckexpect/ambiente/' + str(id_ambiente) + '/'

        code, xml = self.submit(None, 'GET', url)

        key = 'healthcheck_expect'
        return get_list_map(self.response(code, xml, [key]), key)