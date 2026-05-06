def add_healthcheck_expect(self, id_ambiente, expect_string, match_list):
        """Insere um novo healthckeck_expect e retorna o seu identificador.

        :param expect_string: expect_string.
        :param id_ambiente: Identificador do ambiente lógico.
        :param match_list: match list.

        :return: Dicionário com a seguinte estrutura: {'healthcheck_expect': {'id': < id >}}

        :raise InvalidParameterError: O identificador do ambiente, match_lis,expect_string, são inválidos ou nulo.
        :raise HealthCheckExpectJaCadastradoError: Já existe um healthcheck_expect com os mesmos dados cadastrados.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        healthcheck_map = dict()
        healthcheck_map['id_ambiente'] = id_ambiente
        healthcheck_map['expect_string'] = expect_string
        healthcheck_map['match_list'] = match_list

        url = 'healthcheckexpect/add/'

        code, xml = self.submit({'healthcheck': healthcheck_map}, 'POST', url)

        return self.response(code, xml)