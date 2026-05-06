def get_all_rules(self, id_env):
        """Save an environment rule

        :param id_env: Environment id

        :return: Estrutura:

        ::

            { 'rules': [{'id': < id >,
            'environment': < Environment Object >,
            'content': < content >,
            'name': < name >,
            'custom': < custom > },... ]}

        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise UserNotAuthorizedError: Permissão negada.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        url = 'rule/all/' + str(id_env)
        code, xml = self.submit(None, 'GET', url)
        return self.response(code, xml, ['rules'])