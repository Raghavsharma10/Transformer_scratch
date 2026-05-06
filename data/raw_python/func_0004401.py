def configuration_remove(self, environment_id, configuration_id):
        """
        Remove Prefix Configuration

        :return: None

        :raise InvalidValueError: Invalid Id for Environment or IpConfig.
        :raise IPConfigNotFoundError: Ipconfig not resgistred.
        :raise AmbienteNotFoundError: Environment not registered.
        :raise DataBaseError: Failed into networkapi access data base.
        :raise XMLError: Networkapi failed to generate the XML response.

        """

        data = dict()

        data["configuration_id"] = configuration_id
        data["environment_id"] = environment_id

        url = (
            "environment/configuration/remove/%(environment_id)s/%(configuration_id)s/" %
            data)

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)