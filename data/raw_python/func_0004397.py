def update_rule(self, name, id_env, contents, blocks_id, id_rule):
        """
        Save an environment rule

        :param name: Name of the rule
        :param id_env: Environment id
        :param contents: Lists of contents in order. Ex: ['content one', 'content two', ...]
        :param blocks_id: Lists of blocks id or 0 if is as custom content. Ex: ['0', '5', '0' ...]
        :param id_rule: Rule id

        :return: None

        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise InvalidValueError: Invalid parameter.
        :raise UserNotAuthorizedError: Permissão negada.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        url = 'rule/update/'

        map_dict = dict()

        map_dict['name'] = name
        map_dict['id_env'] = id_env
        map_dict['contents'] = contents
        map_dict['blocks_id'] = blocks_id
        map_dict['id_rule'] = id_rule

        try:
            code, xml = self.submit({'map': map_dict}, 'PUT', url)
        except Exception as e:
            raise e

        return self.response(code, xml)