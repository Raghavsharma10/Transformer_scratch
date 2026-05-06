def get_rule_by_pk(self, id_rule):
        """
        Get a rule by its identifier

        :param id_rule: Rule identifier.

        :return: Seguinte estrutura

        ::

            { 'rule': {'id': < id >,
              'environment': < Environment Object >,
              'content': < content >,
              'name': < name >,
              'custom': < custom > }}

        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise InvalidValueError: Invalid parameter.
        :raise UserNotAuthorizedError: Permissão negada.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """
        url = 'rule/get_by_id/' + str(id_rule)
        code, xml = self.submit(None, 'GET', url)
        return self.response(code, xml)