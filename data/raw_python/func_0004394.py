def save_blocks(self, id_env, blocks):
        """
        Save blocks from environment

        :param id_env: Environment id
        :param blocks: Lists of blocks in order. Ex: ['content one', 'content two', ...]

        :return: None

        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise InvalidValueError: Invalid parameter.
        :raise UserNotAuthorizedError: Permissão negada.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        url = 'environment/save_blocks/'

        map_dict = dict()

        map_dict['id_env'] = id_env
        map_dict['blocks'] = blocks

        code, xml = self.submit({'map': map_dict}, 'POST', url)

        return self.response(code, xml)