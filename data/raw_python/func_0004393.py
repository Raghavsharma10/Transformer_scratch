def get_environment_template(self, name, network):
        """Get environments by template name

        :param name: Template name.
        :param network: IPv4 or IPv6.

        :return: Following dictionary:

        ::

            {'ambiente': [divisao_dc - ambiente_logico - grupo_l3, other envs...] }

        :raise InvalidParameterError: Invalid param.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """
        url = 'environment/get_env_template/'

        map_dict = dict()
        map_dict['name'] = name
        map_dict['network'] = network

        code, xml = self.submit({'map': map_dict}, 'PUT', url)

        return self.response(code, xml)