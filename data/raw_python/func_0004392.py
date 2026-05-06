def set_template(self, id_environment, name, network):
        """Set template value. If id_environment = 0, set '' to all environments related with the template name.

        :param id_environment: Environment Identifier.
        :param name: Template Name.
        :param network: IPv4 or IPv6.

        :return: None

        :raise InvalidParameterError: Invalid param.
        :raise AmbienteNaoExisteError: Ambiente não cadastrado.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """

        url = 'environment/set_template/' + str(id_environment) + '/'

        environment_map = dict()
        environment_map['name'] = name
        environment_map['network'] = network

        code, xml = self.submit({'environment': environment_map}, 'POST', url)

        return self.response(code, xml)