def add_ip_range(self, id_environment, id_ip_config):
        """Makes relationship of environment with ip config and returns your id.

        :param id_environment: Environment ID.
        :param id_ip_config: IP Configuration ID.

        :return: Following dictionary:

        {'config_do_ambiente': {'id_config_do_ambiente': < id_config_do_ambiente >}}

        :raise InvalidParameterError: Some parameter was invalid.
        :raise ConfigEnvironmentDuplicateError: Error saving duplicate Environment Configuration.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        environment_map = dict()
        environment_map['id_environment'] = id_environment
        environment_map['id_ip_config'] = id_ip_config

        code, xml = self.submit(
            {'ambiente': environment_map}, 'POST', 'ipconfig/')

        return self.response(code, xml)